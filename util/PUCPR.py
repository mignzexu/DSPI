import os
import pickle

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


class PUCPR(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        subset_scale: float = 1.0,
        resize_val: bool = True,
        args=None,
    ):
        """
        Parameters
        ----------
        data_dir : str, path to the data directory
        split : str, 'train', 'val' or 'test'
        subset_scale : float, scale of the subset of the dataset to use
        resize_val : bool, whether to random crop validation images to 384x384
        anno_file : str, FSC-133 or FSC-147
        """
        assert split in ["train", "val", "test"]

        self.args = args
        #!HARDCODED Dec 25:
        self.data_dir = f"{self.args.data_path}/PUCPR/"

        self.resize_val = resize_val
        self.im_dir = os.path.join(self.data_dir, "Images")
        self.anno_path = os.path.join(self.data_dir, "Annotations")
        self.data_split_path = os.path.join(self.data_dir, "ImageSets")
        self.split = split
        self.split_file = os.path.join(self.data_split_path, split + ".txt")
        with open(self.split_file, "r") as s:
            img_names = s.readlines()
        self.idx_running_set = [x.strip() for x in img_names]
        self.gt_cnt = {}
        self.bbox = {}
        for im_name in self.idx_running_set:
            assert os.path.exists(os.path.join(self.im_dir, f"{im_name}.jpg"))
            assert os.path.exists(os.path.join(self.anno_path, f"{im_name}.txt"))
            with open(os.path.join(self.anno_path, f"{im_name}.txt")) as f:
                boxes = f.readlines()
                # each line is the four coordinates of a bounding box + number of cars in the bounding box
                boxes = [x.strip().split() for x in boxes]
                boxes = [[int(float(x)) for x in box][:4] for box in boxes]
                self.gt_cnt[im_name] = len(boxes)

                self.bbox[im_name] = boxes

        # resize the image height to 384, keep the aspect ratio
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(384),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.idx_running_set)

    def __getitem__(self, idx):
        im_name = self.idx_running_set[idx]
        im_path = os.path.join(self.im_dir, f"{im_name}.jpg")
        img = Image.open(im_path)
        img = self.preprocess(img)
        gt_cnt = self.gt_cnt[im_name]

        return img, gt_cnt


class GPUCPR(PUCPR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gdino_label_dir = os.path.join(self.data_dir, "gdino_label")
        self.gdino_boxes = torch.load(os.path.join(gdino_label_dir, "boxes.pt"))
        self.gdino_logits = torch.load(os.path.join(gdino_label_dir, "logits.pt"))

        if not self.args.online:
            with open(os.path.join(gdino_label_dir, "logits_patches.pkl"), "rb") as f:
                self.gdino_logits_patches = pickle.load(f)

    def __getitem__(self, idx):
        vars_to_return = list(super().__getitem__(idx))
        im_id = self.idx_running_set[idx] + ".jpg"
        # gdino_box = self.gdino_boxes[im_id]
        gdino_logits = self.gdino_logits[im_id].squeeze(dim=0).mean(dim=1)
        # vars_to_return.extend([gdino_box, gdino_logits])
        img_name = os.path.splitext(im_id)[0]

        if not self.args.online:
            gdino_logits_patches = self.gdino_logits_patches[img_name]
            vars_to_return.extend([gdino_logits, img_name, gdino_logits_patches])
        else:
            vars_to_return.extend([gdino_logits, img_name, img_name])

        return vars_to_return
