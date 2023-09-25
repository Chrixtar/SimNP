from abc import ABC, abstractmethod
from copy import deepcopy
import os
import queue
import threading
import tqdm
from typing import Any, Dict, List, Optional, Tuple

from easydict import EasyDict as edict
import numpy as np
import PIL
from PIL.Image import Image

from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import default_collate
from torchvision import transforms
import torchvision.transforms.functional as torchvision_F


class Dataset(Dataset, ABC):

    img_size = None
    dir_name = None

    def __init__(
        self, 
        opt: edict, 
        setting: str,
    ):
        super().__init__()
        assert opt.sizes.image == self.img_size
        self.opt = opt
        self.setting = setting
        self.augmentations = None
        if hasattr(opt.data, self.setting):
            setting_opt = getattr(opt.data, self.setting)
            self.split = getattr(setting_opt, "split", self.setting)
            self.subset = getattr(setting_opt, "sub", None)
            self.n_repeats = getattr(setting_opt, "n_repeats", None)
            if hasattr(setting_opt, "augmentations"):
                self.augmentations = self.init_augmentations(setting_opt.augmentations)
            self.depth = getattr(setting_opt, "load_depth", getattr(opt.data, "load_depth", False))
        else:
            self.split = self.setting
            self.subset = None
            self.n_repeats = None
            self.depth = getattr(opt.data, "load_depth", False)
        
        self.cat = opt.data.cat
        self.path = os.path.join(opt.data.dataset_root_path, self.dir_name)
        self.view_path = self.path
        self.init_objects()
        if self.subset is not None: 
            self.objects = self.objects[:self.subset]
        self.init_views()
        self.init_samples()
        self.preload()

    @abstractmethod
    def init_objects(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def init_views(self) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def init_samples(self) -> None:
        raise NotImplementedError()

    def preprocess_image(
        self, 
        image: Image, 
        mask: Optional[Image] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if mask is not None:
            image = PIL.Image.merge("RGBA", list(image.split())+[mask])
        image = image.resize((self.opt.sizes.image, self.opt.sizes.image))
        image = torchvision_F.to_tensor(image)
        rgb = image[:3]
        if image.shape[0] > 3:
            mask = image[3:]
            rgb *= mask
            if self.opt.data.white_back:
                rgb += 1-mask
        return rgb, mask
    
    def load_image(
        self, 
        img_path: str, 
        mask_path: Optional[str] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        image = PIL.Image.open(img_path).convert("RGB")
        if os.path.isfile(mask_path):
            mask = PIL.Image.open(mask_path).convert("RGB").split()[0]
        else:
            mask = None
        rgb, mask = self.preprocess_image(image, mask)
        return rgb, mask

    @abstractmethod
    def get_image(self, obj_name: str, obj_idx: int, view_name: int) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()
    
    @staticmethod
    def load_pointcloud(
        pc_path: str,
        alignment_path: str
    ) -> Dict[str, Tensor]:
        dpc_numpy = np.load(pc_path)
        dpc = dict(
            points=torch.from_numpy(dpc_numpy["points"]).float(),
            normals=torch.from_numpy(dpc_numpy["normals"]).float(),
        )
        if os.path.isfile(alignment_path):
            alignment = np.load(alignment_path)
            dpc["scale"] = torch.from_numpy(alignment["scale"]).float()
            dpc["offset"] = torch.from_numpy(alignment["offset"]).float()
        return dpc
    
    @abstractmethod
    def get_pointcloud(self, obj_name: str, obj_idx: int) -> Dict[str, Tensor]:
        raise NotImplementedError()

    def parse_intrinsics(self, file_path: str) -> Tensor:
        with open(file_path, 'r') as file:
            f, cx, cy, _ = map(float, file.readline().split())
            next(file)
            next(file)
            height, width = map(float, file.readline().split())
        assert height == width, f"Found non-square camera intrinsics in {file_path}"
        cx = cx / width * self.opt.sizes.render
        cy = cy / height * self.opt.sizes.render
        f = f / height * self.opt.sizes.render
        return torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    @staticmethod
    def parse_extrinsics(file_path: str) -> Tensor:
        pose = np.loadtxt(file_path)
        cam2world = torch.from_numpy(pose).float().view(4, 4)
        # Invert pose (world2cam expected)
        world2cam = cam2world.clone()
        world2cam[:3, :3] = cam2world[:3, :3].transpose(-1, -2)
        world2cam[:3, 3:] = - torch.matmul(world2cam[:3, :3], cam2world[:3, 3:])
        return world2cam

    @abstractmethod
    def get_camera(self, obj_name: str, obj_idx: int, view_name: int) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()
    
    def load_depth(self, path):
        if not os.path.isfile(path):
            return None
        depth = torch.from_numpy(np.load(path)).float().unsqueeze(0)
        if depth.shape[-1] != self.opt.sizes.render:
            depth = F.interpolate(depth.unsqueeze(0), size=self.opt.sizes.render, mode="nearest-exact")[0]
        return depth

    @abstractmethod
    def get_depth(self, obj_name: str, obj_idx: int, view_name: int) -> Tensor:
        raise NotImplementedError()

    def preload_worker(self, data_list, load_func, q, lock, idx_tqdm):
        while True:
            i, idx = q.get()
            data_list[i] = load_func(*idx)
            with lock:
                idx_tqdm.update()
            q.task_done()

    def preload_threading(self, opt, load_func, idx_list, data_str="images"):
        data_list = [None]*len(idx_list)
        q = queue.Queue(maxsize=len(idx_list))
        idx_tqdm = tqdm.tqdm(range(len(idx_list)), desc="preloading {}".format(data_str), leave=False)
        for el in enumerate(idx_list):
            q.put(el)
        lock = threading.Lock()
        for ti in range(opt.data.preload_workers):
            t = threading.Thread(
                target=self.preload_worker, args=(data_list, load_func, q, lock, idx_tqdm), daemon=True)
            t.start()
        q.join()
        idx_tqdm.close()
        # assert(all(map(lambda x: x is not None, data_list)))
        return data_list

    def preload(self):
        # preload dataset
        if self.opt.data.preload or getattr(self.opt.data, "preload_pc", False):
            self.pcs = dict(zip(self.objects, self.preload_threading(self.opt, self.get_pointcloud, self.objects, data_str="pointclouds")))
        if self.opt.data.preload or getattr(self.opt.data, "preload_img", False):
            self.images = dict(zip(self.views, self.preload_threading(self.opt, self.get_image, self.views, data_str="images")))
        if self.opt.data.preload or getattr(self.opt.data, "preload_cam", False):
            self.cameras = dict(zip(self.views, self.preload_threading(self.opt, self.get_camera, self.views, data_str="cameras")))
        if self.depth and (self.opt.data.preload or getattr(self.opt.data, "preload_depth", False)):
            self.depths = dict(zip(self.views, self.preload_threading(self.opt, self.get_depth, self.views, data_str="depths")))

    @staticmethod
    def init_augmentations(aug_configs: List[Dict[str, Any]]):
        aug_configs = deepcopy(aug_configs)
        augmentations = []
        for aug_config in aug_configs:
            aug_name = aug_config.pop("name")
            augmentations.append(getattr(transforms, aug_name)(**aug_config))
        return transforms.Compose(augmentations)

    def setup_loader(self, batch_size, num_workers, shuffle=False, drop_last=True):
        loader = torch.utils.data.DataLoader(
            self, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=shuffle, 
            drop_last=drop_last
        )
        print("number of samples: {}".format(len(self)))
        return loader

    def get_view(self, obj_name: str, obj_idx: int, view_name: int, view_idx: int) -> dict:
        view = {}
        view["img"], mask = self.images[obj_name, obj_idx, view_name] \
            if hasattr(self, "images") else self.get_image(obj_name, obj_idx, view_name)
        if self.augmentations is not None:
            view["img"] = self.augmentations(view["img"])
        if mask is not None:
            view["mask"] = mask
        if self.depth:
            depth = self.depths[obj_name, obj_idx, view_name] \
                if hasattr(self, "depths") else self.get_depth(obj_name, obj_idx, view_name)
            if depth is not None:
                view["depth"] = depth
        view["extr"], view["intr"] = self.cameras[obj_name, obj_idx, view_name] \
            if hasattr(self, "cameras") else self.get_camera(obj_name, obj_idx, view_name)
        view["name"] = view_name
        view["idx"] = view_idx
        return view

    def get_names(self, idx: int):
        n_repeats = self.n_repeats if self.n_repeats is not None else 1
        idx %= len(self.samples) * n_repeats
        sample_idx = idx // n_repeats
        repeat_idx = idx % n_repeats
        sample = self.samples[sample_idx]
        obj_name, obj_idx, views = sample
        return f"{obj_name}_{repeat_idx}" if self.n_repeats is not None else obj_name, \
            [v[0] for v in views] 

    def __getitem__(self, idx: int):
        n_repeats = self.n_repeats if self.n_repeats is not None else 1
        idx %= len(self.samples) * n_repeats
        sample_idx = idx // n_repeats
        repeat_idx = idx % n_repeats
        sample = self.samples[sample_idx]
        obj_name, obj_idx, views = sample
        res = {
            "views": default_collate([self.get_view(obj_name, obj_idx, v, j) for (v, j) in views]),
            "dpc": self.pcs[obj_name, obj_idx] if hasattr(self, "pcs") else self.get_pointcloud(obj_name, obj_idx),
            "obj_idx": obj_idx * n_repeats + repeat_idx,
            "obj_name": f"{obj_name}_{repeat_idx}" if self.n_repeats is not None else obj_name
        }
        return res

    def __len__(self):
        return len(self.samples) \
            * getattr(getattr(self.opt.data, self.setting), "repetitions_per_epoch", 1) \
                * (1 if self.n_repeats is None else self.n_repeats)
