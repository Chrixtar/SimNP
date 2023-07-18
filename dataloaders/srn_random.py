from glob import glob
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import default_collate

from . import base
from utils.util import chunks


class Dataset(base.Dataset):

    def __init__(self, opt, setting, obj_names: List[str], num_random: int = 1):
        super().__init__(opt, setting)
        assert opt.sizes.image == 128
        self.cat = opt.data.cat
        self.path = os.path.join(opt.data.dataset_root_path, "SRN")
        self.init_list_cads()
        if self.subset is not None: 
            self.list_cads = self.list_cads[:self.subset]
        self.obj_name_to_idx = dict(self.list_cads)
        self.obj_names = obj_names
        self.num_random = num_random

        self.intr = self.parse_intrinsics(opt, os.path.join(self.path, "views", "intrinsics.txt"))
        self.extr = torch.stack([self.load_camera(path) for path in sorted(glob(os.path.join(self.path, "views", "test_pose", "*.txt")))], dim=0)
        self.view_idx = range(0, self.extr.shape[0], getattr(opt.data, "view_step_size"))
        self.init_samples()

    def init_list_cads(self):
        cads = []
        blacklist = set()
        whitelist = set()
        if hasattr(self.opt.data, self.setting):
            setting_opt = getattr(self.opt.data, self.setting)
            apply_blacklist = getattr(setting_opt, "blacklist", False)
            if apply_blacklist:
                blacklist_fname = os.path.join(self.opt.data.data_split_path, f"srn_{self.cat}_blacklist.list")
                blacklist = set(open(blacklist_fname).read().splitlines())
                post_blacklist = getattr(setting_opt, "post_blacklist", False)
            else:
                post_blacklist = False
            whitelist = getattr(setting_opt, "whitelist", set())
            post_whitelist = getattr(setting_opt, "post_whitelist", False)
        else:
            post_blacklist = False
            post_whitelist = False
        list_fname = os.path.join(self.opt.data.data_split_path, f"srn_{self.cat}_{self.split}.list")
        i = 0
        for m in open(list_fname).read().splitlines():
            if m not in blacklist and (not whitelist or m in whitelist):
                cads.append((m, i))
                i += 1
            elif post_blacklist or post_whitelist:
                i += 1
        self.list_cads = cads
        self.n_obj = i

    def init_samples(self):
        samples = []
        view_size = self.opt.train_view_size if self.setting == "train" else getattr(self.opt, "vis_view_size", 1)
        for i, m in enumerate(self.obj_names):
            for j in range(self.num_random):
                samples += [(os.path.join(m, str(j)), i * self.num_random + j, v) for v in chunks(self.view_idx, view_size)]
        self.samples = samples

    @staticmethod
    def parse_intrinsics(opt, file_path):
        with open(file_path, 'r') as file:
            f, cx, cy, _ = map(float, file.readline().split())
            next(file)
            next(file)
            height, width = map(float, file.readline().split())
        assert height == width, f"Found non-square camera intrinsics in {file_path}"
        cx = cx / width * opt.sizes.render
        cy = cy / height * opt.sizes.render
        f = f / height * opt.sizes.render
        return torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    @staticmethod
    def load_camera(path):
        pose = np.loadtxt(path)
        cam2world = torch.from_numpy(pose).float().view(4, 4)
        # Invert pose (world2cam expected)
        world2cam = cam2world.clone()
        world2cam[:3, :3] = cam2world[:3, :3].transpose(-1, -2)
        world2cam[:3, 3:] = - torch.matmul(world2cam[:3, :3], cam2world[:3, 3:])
        return world2cam

    def get_view(self, v):
        view = {}
        view["extr"], view["intr"] = self.extr[v], self.intr
        view["view_idx"] = v
        return view

    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        m, i, vs = sample
        res = {
            "views": default_collate([self.get_view(v) for v in vs]),
            "obj_idx": i,
            "obj_name": m
        }
        return res

    def __len__(self):
        return len(self.samples)
