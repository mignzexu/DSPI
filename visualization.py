import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def denormalize_tensor(tensor):
    mean = (
        torch.tensor([0.48145466, 0.4578275, 0.40821073])
        .unsqueeze(1)
        .unsqueeze(1)
        .to(tensor.device)
    )
    std = (
        torch.tensor([0.26862954, 0.26130258, 0.27577711])
        .unsqueeze(1)
        .unsqueeze(1)
        .to(tensor.device)
    )
    denormalized_tensor = tensor * std + mean
    return denormalized_tensor


def vis_norm(var, flag=False):
    var = (var - np.min(var)) / (np.max(var) - np.min(var))
    if flag:
        var = 1.0 - var
    var = var * 255.0
    var = np.uint8(var)

    return var


def image2RGB(image):
    image = denormalize_tensor(image)
    image = image.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
    image = image.detach().cpu().numpy()
    image = vis_norm(image, False)
    image = Image.fromarray(image, mode="RGB")
    return image


def vector2patch(vector):  # [HW, C] -> [C, H, W]
    hw, c = vector.shape
    h = w = int(math.sqrt(hw))
    patch = vector.transpose(0, 1).reshape(c, h, w)

    return patch


def feature2RGB(feature, category="mean", width=224, height=224, flag=False):
    if feature.dim() == 2:
        feature = vector2patch(feature)  # [C, H, W]

    if category == "mean":
        feature = feature.mean(0)
    elif category == "max":
        feature = feature.max(0)[0]
    elif category == "min":
        feature = feature.min(0)[0]
    else:
        raise NotImplementedError

    feature = feature.detach().cpu().numpy()
    feature = vis_norm(feature, flag)
    image = Image.fromarray(feature).resize((width, height))

    return image


def heatmap(img, map1, map2, map3, map4, map5, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)

    image = image2RGB(img)
    width, height = image.size

    heatmap1 = feature2RGB(map1, "mean", width, height, True)
    heatmap2 = feature2RGB(map2, "mean", width, height, True)
    heatmap3 = feature2RGB(map3, "mean", width, height, False)
    heatmap4 = feature2RGB(map4, "mean", width, height, False)
    heatmap5 = feature2RGB(map5, "mean", width, height, False)

    figure, axes = plt.subplots(1, 6)

    axes[0].imshow(image)
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(heatmap1, alpha=0.6, cmap="jet")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(heatmap2, alpha=0.6, cmap="jet")
    axes[2].axis("off")

    axes[3].imshow(image)
    axes[3].imshow(heatmap3, alpha=0.6, cmap="jet")
    axes[3].axis("off")

    axes[4].imshow(image)
    axes[4].imshow(heatmap4, alpha=0.6, cmap="jet")
    axes[4].axis("off")

    axes[5].imshow(image)
    axes[5].imshow(heatmap5, alpha=0.6, cmap="jet")
    axes[5].axis("off")

    figure.set_size_inches(width * 6 / 300, height / 300)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(os.path.join(save_dir, name), dpi=300)
    plt.close()