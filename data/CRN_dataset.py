from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import h5py
import random
import numpy as np

class CRNShapeNet(data.Dataset):
    """
    Dataset with GT and partial shapes provided by CRN
    Used for shape completion and pre-training tree-GAN
    """
    def __init__(self, args):
        self.args = args
        self.dataset_path = self.args.dataset_path
        self.class_choice = self.args.class_choice
        self.split = self.args.split

        pathname = os.path.join(self.dataset_path, f'{self.split}_data.h5')
        
        data = h5py.File(pathname, 'r')
        self.gt = data['complete_pcds'][()]
        self.partial = data['incomplete_pcds'][()]
        self.labels = data['labels'][()]
        
        np.random.seed(0)
        cat_ordered_list = ['plane','cabinet','car','chair','lamp','couch','table','watercraft']

        cat_id = cat_ordered_list.index(self.class_choice.lower())
        self.index_list = np.array([i for (i, j) in enumerate(self.labels) if j == cat_id ])                      

    def __getitem__(self, index):
        full_idx = self.index_list[index]
        gt = torch.from_numpy(self.gt[full_idx]) # fast alr
        label = self.labels[index]
        partial = torch.from_numpy(self.partial[full_idx])
        return gt, partial, full_idx

    def __len__(self):
        return len(self.index_list)

class KITTI_loader(data.Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, args):
        super(KITTI_loader, self).__init__()
        self.args = args
        self.dataset_path = self.args.dataset_path
        npyfile = np.load(self.dataset_path)
        dataxyz = self.from_polar_np(npyfile)
        self.dataset = dataxyz.reshape(-1,2048,3)
        # self.complete = np.concatenate([self.from_polar_np(np.load(self.dataset_path, mmap_mode='r')[:, :, :, ::8]) for i in range(2)], axis=0).transpose(0, 2, 3, 1).reshape(-1, 2048, 3)
        print(self.dataset.shape)

    def __len__(self):
        # batchsize
        return self.dataset.shape[0]

    def __getitem__(self, index):
        
        return self.dataset[index]

    def from_polar_np(self, velo):
        angles = np.linspace(0, np.pi * 2, velo.shape[-1])
        dist, z = velo[:, 0], velo[:, 1]
        x = np.cos(angles) * dist
        y = np.sin(angles) * dist
        out = np.stack([x,y,z], axis=1)
        return out.astype('float32')
