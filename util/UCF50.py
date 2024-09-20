import json
import os
import pickle

import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms

IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


class UCF50(Dataset):
    def __init__(self, split, part, args=None):
        assert split in ["train", "test"]
        assert part in ["0", "1", "2", "3", "4", "5"]

        self.split = split
        self.part = part
        self.args = args
        self.data_dir = f"{self.args.data_path}/UCF50"

        with open(os.path.join(self.data_dir, "all_data.json"), "r") as f:
            self.all_data = json.load(f)
        self.img_names = self.all_data[self.part]

        self.gt_cnt = {}

        for im_name in self.img_names:
            img_path = os.path.join(self.data_dir, im_name)
            assert os.path.exists(img_path)
            gt_path = img_path.replace(".jpg", "_ann.mat")

            with open(gt_path, "rb") as f:
                mat = loadmat(f)
                # the number of count is lenth of the points
                self.gt_cnt[im_name] = len(mat["annPoints"])

        # resize the image height to 384, keep the aspect ratio
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(384),
                transforms.ToTensor(),
            ]
        )

        gdino_label_dir = os.path.join(self.data_dir, "gdino_label")
        self.gdino_boxes = torch.load(
            os.path.join(gdino_label_dir, "boxes.pt"), map_location=torch.device("cpu")
        )
        self.gdino_logits = torch.load(
            os.path.join(gdino_label_dir, "logits.pt"), map_location=torch.device("cpu")
        )
        if not self.args.online:
            with open(os.path.join(gdino_label_dir, "logits_patches.pkl"), "rb") as f:
                self.gdino_logits_patches = pickle.load(f)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        im_name = self.img_names[idx]
        im_path = os.path.join(self.data_dir, im_name)
        img = Image.open(im_path)
        # if the image height larger than width, rotate it
        if img.size[0] < img.size[1]:
            img = img.rotate(90, expand=True)
        # if the image is grayscale, convert it to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self.preprocess(img)
        gt_cnt = self.gt_cnt[im_name]

        gdino_logits = self.gdino_logits[im_name].squeeze(dim=0).mean(dim=1)

        if not self.args.online:
            img_name = os.path.splitext(im_name)[0]
            gdino_logits_patches = self.gdino_logits_patches[img_name]
            return img, gt_cnt, gdino_logits, img_name, gdino_logits_patches
                     
        img_name = os.path.splitext(im_name)[0]
        return img, gt_cnt, gdino_logits, img_name, img_name
