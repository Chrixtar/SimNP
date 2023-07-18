from glob import glob
import os
from typing import List, Tuple

import torch
from torch.utils.data import default_collate

from .srn import SRN
from utils.util import chunks


class SRNInterpolation(SRN):

    def __init__(
        self, 
        opt, 
        setting: str,
        obj_name_pairs: List[Tuple[int, int]], 
        num_steps: int
    ):
        self.obj_name_pairs = obj_name_pairs
        self.num_steps = num_steps
        super().__init__(opt, setting)
        self.obj_name_to_idx = dict(self.objects)

    def preload(self):
        self.intr = self.parse_intrinsics(os.path.join(self.path, "views", "intrinsics.txt"))
        self.extr = torch.stack([self.parse_extrinsics(path) for path in sorted(glob(os.path.join(self.path, "views", "test_pose", "*.txt")))], dim=0)

    def init_views(self):
        num_poses = len(glob(os.path.join(self.path, "views", "test_pose", "*.txt")))
        self.view_idx = range(0, num_poses, getattr(self.opt.data, "view_step_size"))
    
    def init_samples(self):
        samples = []
        view_size = self.opt.train_view_size if self.setting == "train" else getattr(self.opt, "vis_view_size", 1)
        for i, (m1, m2) in enumerate(self.obj_name_pairs):
            for j in range(self.num_steps):
                samples += [(os.path.join(f"{m1}_{m2}", str(j)), i * self.num_steps + j, v) for v in chunks(self.view_idx, view_size)]
        self.samples = samples

    def get_view(self, v):
        view = {}
        view["extr"], view["intr"] = self.extr[v], self.intr
        view["name"] = v
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
