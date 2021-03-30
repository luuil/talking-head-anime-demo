# coding: utf-8
# Created by luuil@outlook.com at 3/30/2021
import os

from torch.utils.data import Dataset
from torchvision import transforms
from util import pkl_load, read_pil_img_rgb
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np


class EyeDataset(Dataset):
    def __init__(self, pkl_file, root_dir, is_train=True, debug=False):
        self.root_dir = root_dir
        self.debug = debug
        self.is_train = is_train
        img_pose_pairs = pkl_load(pkl_file)

        pose_channels = 9
        samples_per_pose = 100
        samples_per_img = pose_channels * samples_per_pose + 1  # 1: in rest pose
        self.expressive_samples_per_img = pose_channels * samples_per_pose

        rest_pose_imgs = list()
        cnt = 0
        for name in img_pose_pairs:
            if cnt % samples_per_img == 0:
                rest_pose_imgs.append(name)
            cnt += 1

        # remove samples in rest pose
        for rest_img_name in rest_pose_imgs:
            del img_pose_pairs[rest_img_name]

        self.size = len(img_pose_pairs)
        self.rest_pose_imgs = rest_pose_imgs
        self.expressive_img_pose_pairs = img_pose_pairs

    def __getitem__(self, index):
        expressive_imgs = list(self.expressive_img_pose_pairs)
        expressive_img_name = expressive_imgs[index]
        expressive_img_path = os.path.join(self.root_dir, expressive_img_name)
        rest_img_name = self.rest_pose_imgs[index // self.expressive_samples_per_img]
        rest_img_path = os.path.join(self.root_dir, rest_img_name)
        pose = self.expressive_img_pose_pairs[expressive_img_name]
        img_r = read_pil_img_rgb(rest_img_path)
        img_e = read_pil_img_rgb(expressive_img_path)
        img_r, img_e = self._transform(img_r, img_e)
        sample = {
            "rest": img_r,
            "expressive": img_e,
            "pose": pose,
        }
        if self.debug:
            sample["rest_name"] = rest_img_name
            sample["expressive_name"] = expressive_img_name

        return sample

    def __len__(self):
        return self.size

    def _transform(self, img_r, img_e, sz=64, flip=0.5, interp=Image.BICUBIC):
        # Resize
        resize = transforms.Resize(sz, interpolation=interp)
        img_r = resize(img_r)
        img_e = resize(img_e)

        # Random horizontal flipping, only for train
        if self.is_train:
            if np.random.random() > flip:
                img_r = TF.hflip(img_r)
                img_e = TF.hflip(img_e)

        if not self.debug:
            transforms_list = [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ]
            composed = transforms.Compose(transforms_list)
            img_r = composed(img_r)
            img_e = composed(img_e)

        return img_r, img_e


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    eye_dataset = EyeDataset(
        pkl_file=r'G:\projects\0results\cartoon_controllable\waifu512_n100\crop_left_eye.pkl',
        root_dir=r'G:\projects\0results\cartoon_controllable\waifu512_n100\crop_left_eye',
        debug=True
    )
    selected_sample = eye_dataset[203]
    fig = plt.figure()

    ax = plt.subplot(1, 2, 1)
    plt.tight_layout()
    ax.set_title(selected_sample["rest_name"])
    plt.imshow(selected_sample["rest"])

    ax = plt.subplot(1, 2, 2)
    plt.tight_layout()
    ax.set_title(selected_sample["expressive_name"])
    plt.imshow(np.asarray(selected_sample["expressive"]))

    plt.show()

    print(f'pose={selected_sample["pose"]}')
