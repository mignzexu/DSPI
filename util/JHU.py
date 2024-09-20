import glob
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


class JHU(Dataset):
    def __init__(self, split, part=None, args=None):
        assert split in ["train", "test"]

        self.split = split
        self.args = args
        self.data_dir = f"{self.args.data_path}/JHU/{split}"

        self.img_paths = glob.glob(os.path.join(self.data_dir, "images", "*.jpg"))
        self.gt_cnt = {}

        for img_path in self.img_paths:
            assert os.path.exists(img_path)
            img_name = os.path.basename(img_path)
            gt_path = img_path.replace(".jpg", ".txt").replace("images", "gt")

            with open(gt_path, "r") as file:
                lines = file.readlines()
                # the number of count is lenth of the points
                self.gt_cnt[img_name] = len(lines)

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
        return len(self.img_paths)

    def __getitem__(self, idx):
        im_path = self.img_paths[idx]
        im_name = os.path.basename(im_path)
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
