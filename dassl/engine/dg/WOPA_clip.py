import datetime
import time
import torch
import os
from tqdm import tqdm
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import clip
from collections import OrderedDict
from dassl.data import DataManager,DataManager_sf
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.modeling import build_head, build_backbone, Backbone
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import AngularPenaltySMLoss,EntropyMaximization,InfoNCE
from dassl.modeling.head import se_attn_sr
from dassl.evaluation import build_evaluator


class clip_net(nn.Module):
    def __init__(self, cfg, model_cfg, device,**kwargs):
        super().__init__()

        self.device=device
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            device=self.device,
            **kwargs
        )
        self.fdim=self.backbone._out_features
        self.head = None
        if model_cfg.HEAD.NAME=='se_attn_sr':
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                num_channels=self.fdim,
                reduction_ratio=model_cfg.HEAD.REDUCTION_RATIO,
                **kwargs,
            )
            self.fdim = self.head.out_features


    def forward_text(self, x,t_x):
        t = self.backbone.forward_text(x,t_x)  #text embed without norm
        t=t/t.norm(dim=-1,keepdim=True)   #norm after embed
        if self.head is not None:
            t=self.head(t)     #text embed after head without norm
        return t
    
    def forward_img(self,x):  #for test
        t_img=self.backbone.forward_image(x)
        t_img=t_img/t_img.norm(dim=-1,keepdim=True)  #norm after embed
        if self.head is not None:
            t_img=self.head(t_img)     #img embed after head without norm
        return t_img

class clip_net_arcface(nn.Module):
    def __init__(self, cfg, model_cfg, num_classes, device, loss_type='arcface'):
        super(clip_net_arcface, self).__init__()
        self.embedlayers = clip_net(cfg, model_cfg, device)
        in_features=self.embedlayers.fdim  #embed dim
        self.classifier = nn.Linear(in_features, num_classes)  # 新增的分类层

    def forward_img(self, text, img, norm=False):
        img_feature = self.embedlayers.backbone.forward(img, text)
        if norm:
            img_feature = img_feature / img_feature.norm(dim=-1, keepdim=True)  # normalize

        return img_feature

    def predictor(self,feat,teat):
        feat_p = feat/feat.norm(dim=-1,keepdim=True)
        feat_p = feat_p.half()
        teat_p = teat/teat.norm(dim=-1, keepdim=True)
        teat_p = teat_p.half()
        scores =  (100.0 * torch.matmul(feat_p,teat_p.detach().T))
        scores = torch.cat([scores,torch.zeros(scores.shape[0],1,device=scores.device)],1)
        return scores

    
@TRAINER_REGISTRY.register()
class WOPA_clip(TrainerX):
    def __init__(self,cfg):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device(f"cuda:{cfg.DEVICE}")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        self.cfg = cfg
        
        self.build_model()
        self.build_train_data()
        self.build_data_loader()
        
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf

        self.enm_loss = EntropyMaximization()
        self.infonce_loss=InfoNCE(negative_mode='paired')

    def build_model(self):
        cfg = self.cfg
        print("Building model")

        if self.cfg.DATASET.NAME=='PACS_SF':
            self.num_classes=7
        elif self.cfg.DATASET.NAME=='OfficeHomeDG_SF':
            self.num_classes=65
        elif self.cfg.DATASET.NAME=='VLCS_SF':
            self.num_classes=5
        elif self.cfg.DATASET.NAME=='DomainNet_SF':
            self.num_classes=345

        self.model = clip_net_arcface(cfg, cfg.MODEL, self.num_classes,self.device)
        # self.model = clip_softmax(cfg, cfg.MODEL, self.num_classes, self.device)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        trainable_param = self.model.embedlayers.backbone.get_trainable_params()
        self.optim = build_optimizer(self.model, cfg.OPTIM, param_groups=trainable_param)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.criterion = nn.CrossEntropyLoss()
        self.register_model("model", self.model, self.optim, self.sched)

    def forward_backward(self, batch):
        label, domain, img = self.parse_batch_train(batch)
        cls_txt_weight = self.prompt_generater.cls_text_weight
        img_feature = self.model.forward_img(cls_txt_weight, img, norm=True)
        similarities = self.model.predictor(img_feature, cls_txt_weight)

        loss = self.criterion(similarities, label)

        self.model_backward_and_update(loss)

        _, predicted = torch.max(similarities.data, 1)
        accuracy = (predicted == label).sum().item() / label.size(0)


        loss_summary = {
            "loss": loss.item(),
            "acc": accuracy,
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        label = batch["label"]
        domain = batch["domain"]
        img = batch["img"]

        label = label.to(self.device)
        domain = domain.to(self.device)
        img = img.to(self.device)
        return label, domain, img


    def build_data_loader(self):
        train_data=None
        dm = DataManager_sf(self.cfg,train_data)

        self.train_loader_x = dm.train_loader_x
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}
        self.dm = dm
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        result=[]
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader
        for data_loader_domain in data_loader:
            print(f"Evaluate on the *{split}* set")
            for batch_idx, batch in enumerate(tqdm(data_loader_domain)):
                input, label = self.parse_batch_test(batch)
                output = self.model_inference_vit(input) # zero_shot
                # output = self.model_inference_vit_not_val(input)
                self.evaluator.process(output, label)

            results = self.evaluator.evaluate()

            for k, v in results.items():
                tag = f"{split}/{k}"
                self.write_scalar(tag, v, self.epoch)
            result.append(list(results.values())[0])
            self.evaluator.reset()

        return result

    def after_epoch(self):
        self.test()
        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")


    def train(self):
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def model_inference(self, input):
        cls_txt_weight=self.prompt_generater.cls_text_weight
        img_feat=self.model.embedlayers.backbone.forward_image(input)
        scores=self.model.predictor(img_feat,cls_txt_weight)
        return scores

    def model_inference_vit(self, input):
        cls_txt_weight=self.prompt_generater.cls_text_weight
        input = input.half()
        img_feat=self.model.embedlayers.backbone.forward(input, cls_txt_weight)
        scores=self.model.predictor(img_feat,cls_txt_weight)
        return scores

    def build_train_data(self):
        
        txts_dir_path=self.cfg.TXTS_PATH
        txt_path=os.path.join(txts_dir_path,self.cfg.DATASET.NAME+'.txt')
        
        with open(txt_path,'r') as f:
            lines = f.read().splitlines()
        class_dict = {index: value for index, value in enumerate(lines)}
        classnames=list(class_dict.values())
        self.num_classes=len(classnames)
        self.prompt_generater=PromptLearner(classnames,self.model.embedlayers.backbone,self.device)


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, device):
        super().__init__()
        self.classnames = classnames
        self.device = device
        dtype = clip_model.dtype

        self.cls_prompt, self.cls_embedding = self.cls_text_embedding(clip_model, dtype, classnames)
        self.cls_text_weight = clip_model.forward_text(self.cls_embedding, self.cls_prompt)

    def cls_text_embedding(self, clip_model, dtype, cls_names):
        cls_list = ["a photo of a " + s for s in cls_names]
        cls_prompt = clip.tokenize(cls_list).to(self.device)
        with torch.no_grad():
            cls_embedding = clip_model.token_embedding(cls_prompt).type(dtype)
        return cls_prompt, cls_embedding



