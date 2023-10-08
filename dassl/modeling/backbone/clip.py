from clip.model import *
from clip.clip import load
import torch
from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from ..head.Cross_attention import NLBlockND_cross


class ResNet_clip(Backbone):
    def __init__(self, clip_enocder_name,out_dim,device):
        super().__init__()
        self.model, self.preprocess = load(clip_enocder_name,device=device)

        self.model.float()
        self.device=device
        # freeze everything
        for name, val in self.model.named_parameters():
            val.requires_grad = False
        #image part
        self._out_features = out_dim
        #text part 
        self.transformer = self.model.transformer
        self.positional_embedding = self.model.positional_embedding
        self.ln_final = self.model.ln_final
        self.text_projection = self.model.text_projection
        self.dtype = self.model.dtype
        self.token_embedding = self.model.token_embedding

    def forward_text(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x).type(self.dtype)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
    
    def forward_image(self, x):
        image_features = self.model.encode_image(x)
        return image_features


class Modified_VitClip(ResNet_clip):
    def __init__(self, clip_enocder_name, out_dim, device):
        super(Modified_VitClip, self).__init__(clip_enocder_name, out_dim, device)
        self.model, self.preprocess = load(clip_enocder_name, device=device, modified=True)
        for param in self.model.parameters():
            param.requires_grad = False

        # self.cross_attention = NLBlockND_cross(in_channels=out_dim, mode='embedded',
        #                                        dimension=2)  # Assuming 2D features
        self.cross_attention = NLBlockND_cross(in_channels=768, mode='embedded')

        for param in self.cross_attention.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        return [p for p in self.cross_attention.parameters() if p.requires_grad]


    def forward(self, x, text_embedding):
        # Obtain features up to the mid of the vision encoder
        mid_features = self.model.visual.encode_to_mid(x)#[197, 32, 768] [序列长度+1,批大小,维度]

        # Here, get the text output using the transformer from ResNet_clip
        # text_features = self.forward_text(text_embedding)

        # Perform cross attention between mid_features and text_features
        # attended_features = self.cross_attention(mid_features, text_embedding)

        attended_features = self.cross_attention(mid_features, mid_features)#strong basline
        attended_features = attended_features.permute(2, 0, 1).float() # Change shape from [32, 768, 197] to [197, 32, 768]

        # attended_features  = attended_features + mid_features
        # Pass through the second half of the vision encoder
        image_features = self.model.visual.encode_from_mid(attended_features)

        return image_features

    def forward_text(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x).type(self.dtype)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x



@BACKBONE_REGISTRY.register()
def resnet50_clip(device,**kwargs):
    model=ResNet_clip('RN50',1024,device)
    return model


@BACKBONE_REGISTRY.register()
def vitb16_clip(device, **kwargs):
    model = ResNet_clip('ViT-B/16', 1024, device)
    return model

@BACKBONE_REGISTRY.register()
def modified_vitb16_clip(device, **kwargs):
    model = Modified_VitClip('ViT-B/16', 512, device)

    return model
