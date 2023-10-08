import glob
import os.path as osp
from dassl.utils import listdir_nohidden
from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase,Datum_sf
from .digits_dg import DigitsDG

@DATASET_REGISTRY.register()
class OfficeHomeDG_SF(DatasetBase):
    """Office-Home.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """

    dataset_dir = "office_home_dg"
    domains = ["none","art", "clipart", "product", "real_world"]
    data_url = "https://drive.google.com/uc?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa"

    def __init__(self, cfg, train_data):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)



        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "office_home_dg.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )
        train_datasets = []
        for domain in cfg.DATASET.TARGET_DOMAINS:
            train_datasets.append(DigitsDG.read_data(self.dataset_dir, [domain], "train"))

        test_datasets = []
        for domain in cfg.DATASET.TARGET_DOMAINS:
            test_datasets.append(DigitsDG.read_data(self.dataset_dir,[domain], "val"))

        super().__init__(train_x=train_datasets, test=test_datasets)

