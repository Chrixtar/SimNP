import os
import pickle
from typing import Dict, Tuple

from torch import Tensor

from .base import Dataset
from utils.util import chunks


class SRN(Dataset):

    img_size = 128
    dir_name = "SRN"

    def __init__(
        self, 
        opt, 
        setting: str = "train",
    ):
        super().__init__(opt, setting)

    def init_objects(self):
        objects = []
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
                objects.append((m, i))
                i += 1
            elif post_blacklist or post_whitelist:
                i += 1
        self.objects = objects
        self.n_obj = i
        if self.n_repeats is not None:
            self.n_obj *= self.n_repeats

    def init_views(self):
        with open(os.path.join(self.opt.data.data_split_path, f"srn_{self.cat}_view.pkl"), "rb") as file:
            view_idx = pickle.load(file)
        setting_opt = getattr(self.opt.data, self.setting)
        view_size = self.opt.train_view_size if self.setting == "train" else getattr(self.opt, "vis_view_size", 1)
        view_slices = getattr(setting_opt, f"view_slices", [[0, 50]] if self.split == "train" else [[0, 251]])
        self.view_idx = [i for s in view_slices for i in range(*s)]
        assert len(self.view_idx) % view_size == 0 or len(self.view_idx) < view_size, f"Number of views {len(self.view_idx)} must be divisible by or smaller than view_size {view_size}! (otherwise dimensionality problems in collate)"
        self.object_views = {}
        self.views = set()
        for m, i in self.objects:
            self.object_views[m, i] = []
            for j, view in enumerate(self.view_idx):
                view = view_idx[m][view]
                self.object_views[m, i].append((view, j))
                self.views.add((m, i, view))

    def init_samples(self):
        samples = []
        view_size = self.opt.train_view_size if self.setting == "train" else getattr(self.opt, "vis_view_size", 1)
        for (m, i), views in self.object_views.items():
            # shuffle(views)
            samples += [(m, i, v) for v in chunks(views, view_size)]
        self.samples = samples

    def get_image(self, obj_name: str, obj_idx: int, view_name: int) -> Tuple[Tensor, Tensor]:
        path = os.path.join(self.view_path, self.cat, obj_name)
        return self.load_image(
            os.path.join(path, "rgb", f"{view_name:06}.png"),
            os.path.join(path, "segmentation", f"{view_name:06}.png")
        )
    
    def get_pointcloud(self, obj_name: str, obj_idx: int) -> Dict[str, Tensor]:
        path = os.path.join(self.view_path, self.cat, obj_name)
        return self.load_pointcloud(
            os.path.join(path, "pointcloud3.npz"),
            os.path.join(path, "pointcloud_alignment_params.npz")
        )

    def get_camera(self, obj_name: str, obj_idx: int, view_name: int) -> Tuple[Tensor, Tensor]:
        path = os.path.join(self.view_path, self.cat, obj_name)
        extr = self.parse_extrinsics(os.path.join(path, "pose", f"{view_name:06}.txt"))
        intr = self.parse_intrinsics(os.path.join(path, "intrinsics.txt"))
        return extr, intr

    def get_depth(self, obj_name: str, obj_idx: int, view_name: int) -> Tensor:
        depth_dir = "pred_depth" if hasattr(self.opt.data, self.setting) and getattr(getattr(self.opt.data, self.setting), "pred_depth", False) else "depth"
        path = os.path.join(self.view_path, self.cat, obj_name, depth_dir, f"{view_name:06}.npy")
        return self.load_depth(path)
