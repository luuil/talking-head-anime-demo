# coding: utf-8
# Created by luuil@outlook.com at 3/30/2021

from torch.utils import data
from dataset.eye_dataset import EyeDataset


def load_dataset(pkl_file,
                 root_dir,
                 is_train,
                 batch_size,
                 shuffle,
                 num_workers):
    dataset = EyeDataset(pkl_file=pkl_file,
                         root_dir=root_dir,
                         is_train=is_train)
    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           drop_last=True)
