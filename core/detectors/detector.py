from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from easydict import EasyDict as edict
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from . import priors
from utils.pinned_embedding import PinnedEmbedding


class Detector(Module, ABC):
    def __init__(
        self, 
        in_dim: int,
        hid_dim: int,
        cube_scale: float,
        n_obj: int,
        n_keypoints: int,
        predict_lrf: bool = False,
        prior: Optional[edict] = None,
        scale_emb: bool = False,
        offset_emb: bool = False,
        scale_emb_gpu: bool = True,
        offset_emb_gpu: bool = True,
        center_canonical: bool = True,
        predict_mirror: bool = False,
        use_cache: bool = False
    ) -> None:
        super(Detector, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.cube_scale = cube_scale
        self.n_keypoints = n_keypoints
        self.n_obj = n_obj
        self.predict_lrf = predict_lrf
        self.center_canonical = center_canonical
        self.predict_mirror = predict_mirror
        self.use_cache = use_cache
        if prior is not None:
            self.prior = getattr(priors, prior.network)(self.hid_dim, **prior.kwargs)
        else:
            self.prior = None
        self.has_scale_emb = scale_emb
        if self.has_scale_emb:
            self.scale_emb = PinnedEmbedding(self.n_obj, 1, scale_emb_gpu, flex=True)
        self.has_offset_emb = offset_emb
        if self.has_offset_emb:
            self.offset_emb = PinnedEmbedding(self.n_obj, 3, offset_emb_gpu, flex=True)
        if self.use_cache:
            self.register_buffer("cache_mask", torch.zeros(n_obj, dtype=torch.bool), persistent=False)
            self.register_buffer("pos_cache", torch.empty((n_obj, n_keypoints, 3)), persistent=False)
            if self.predict_lrf:
                self.register_buffer("rot_cache", torch.empty((n_obj, n_keypoints, 4)), persistent=False)
                self.register_buffer("mir_cache", torch.empty((n_obj, n_keypoints, 1)), persistent=False)

    def interpolate(self, idx: Tensor, num_steps: int) -> None:
        """
        Arguments
            idx: [n, 2]
        """
        raise NotImplementedError()
    
    def randomize(self, idx: Tensor, num_random: int, std: float) -> None:
        """
        Arguments
            idx: [n]
        """
        raise NotImplementedError()

    def copy_selection(self, idx: Tensor, num_copies: int) -> None:
        """
        Arguments
            idx: [n]
        """
        raise NotImplementedError()

    @abstractmethod
    def encode(self, idx: Tensor) -> Tensor:
        """
        Arguments:
            idx: [B]
        Returns:
            latent: [B, hid_dim]
        """
        raise NotImplementedError()

    def prior_encode(
        self, 
        img: Tensor, 
        mask: Optional[Tensor],
        pix: Optional[Tensor],
        extr: Optional[Tensor], 
        intr: Optional[Tensor]
    ) -> Tensor:
        """
        Arguments:
            img: [B, num_views, in_dim, H, W]
            optional mask: [B, num_views, 1, H, W]
            optional pix: [B, num_views, 2, H, W]
            optional extr: [B, num_views, 4, 4]
            optional intr: [B, num_views, 4, 4]
        Return:
            latent: [B, num_views, hid_dim]
        """
        B, num_views = img.shape[:2]
        if mask is not None:
            mask = mask.flatten(0, 1)
        if pix is not None:
            pix = pix.flatten(0, 1)
        if extr is not None:
            extr = extr.flatten(0, 1)
            intr = intr.flatten(0, 1)
        latent = self.prior(
            img.flatten(0, 1),
            mask,
            pix,
            extr,
            intr
        ).view(B, num_views, -1)
        return latent

    @abstractmethod
    def decode(self, latent: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Arguments:
            latent: [L, hid_dim]
        Returns:
            pos: [L, n_kp, 3]
            optional lrf: [L, n_kp, 5]
        """
        raise NotImplementedError()
    
    def is_cached(self, idx: Tensor) -> bool:
        """
        Arguments:
            idx: [B]
        """
        return self.cache_mask[idx].all()

    def get_cached(self, idx: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Arguments:
            idx: [B]
        Returns:
            pos: [B, n_keypoints, 3]
            optional rot: [B, n_keypoints, 4]
            optional mir: [B, n_keypoints, 1]
        """
        pos = self.pos_cache[idx]
        if self.predict_lrf:
            rot = self.rot_cache[idx]
            mir = self.mir_cache[idx]
        else:
            rot = mir = None
        return pos, rot, mir

    def set_cache(self, idx: Tensor, pos: Tensor, rot: Optional[Tensor], mir: Optional[Tensor]) -> None:
        """
        Arguments:
            idx: [B]
            pos: [B, n_keypoints, 3]
            optional rot: [B, n_keypoints, 4]
            optional mir: [B, n_keypoints, 1]
        """
        self.pos_cache[idx] = pos.detach()
        if self.predict_lrf:
            self.rot_cache[idx] = rot.detach()
            self.mir_cache[idx] = mir.detach()
        self.cache_mask[idx] = True

    def forward(
        self, 
        idx: Tensor, 
        img: Optional[Tensor],
        mask: Optional[Tensor],
        pix: Optional[Tensor],
        extr: Optional[Tensor],
        intr: Optional[Tensor],
        scale: Optional[Tensor],
        offset: Optional[Tensor],
        apply_scale: bool = True,
        apply_offset: bool = True,
        return_kp_pos: bool = True, 
        return_prior: bool = False,
        return_latent: bool = False
) -> Dict[str, Tensor]:
        """
        Arguments:
            idx: [B]
            optional img: [B, num_views, in_dim, H, W]
            optional mask: [B, num_views, 1, H, W]
            optional pix: [B, num_views, 2, H, W]
            optional extr: [B, num_views, 4, 4]
            optional intr: [B, num_views, 3, 3]
            optional scale: [B, 1]
            optional offset: [B, 3]
        Returns: Dictionary with
            if return_kp_pos:
                canonical_kp_pos: [B, num_kp, 3]
                kp_pos: [B, num_kp, 3]
                if self.predict_lrf:
                    kp_rot: [B, num_kp, 4]
                    kp_mir: [B, num_kp, 1]
            if return_prior:
                canonical_prior_kp_pos: [B, num_views, num_kp, 3]
                prior_kp_pos [B, num_views, num_kp, 3]
                if self.predict_lrf:
                    prior_kp_rot: [B, num_views, num_kp, 4]
                    prior_kp_mir: [B, num_views, num_kp, 1]
            if return_latent:
                if return_kp_pos:
                    kp_latent: [B, hid_dim]
                if return_prior:
                    prior_kp_latent: [B, num_views, hid_dim]
        """
        assert return_kp_pos or return_prior, "Return at least one of the two point clouds"
        assert not self.use_cache or not return_latent, "Cannot return latent while using cache"
        B = len(idx)
        num_views = img.shape[1] if img is not None else 1
        device = idx.device
        res = {}
        is_cached = self.use_cache and self.is_cached(idx)
        if return_kp_pos and not is_cached:
            latent = self.encode(idx)   # [B, hid_dim]
            if return_latent:
                res["kp_latent"] = latent
        else:
            latent = torch.empty((0, self.hid_dim), device=device)
        if return_prior:
            prior_latent = self.prior_encode(img, mask, pix, extr, intr).flatten(0, 1)
            if return_latent:
                res["prior_kp_latent"] = prior_latent.view(B, num_views, -1)
            latent = torch.cat((latent, prior_latent), dim=0)
        if latent.numel() > 0:
            pos, lrf = self.decode(latent)
            kp_pos = self.cube_scale * torch.tanh(pos)
            if self.predict_lrf:
                kp_rot = F.normalize(lrf[..., :-1], dim=-1)
                kp_mir = torch.sigmoid(lrf[..., -1:])
        else:
            kp_pos = kp_rot = kp_mir = None
        if apply_scale and self.has_scale_emb and scale is None:
            scale = self.scale_emb(idx) # [B, 1]
        if apply_offset and self.has_offset_emb and offset is None:
            offset = self.offset_emb(idx)  # [B, 3]
        if return_kp_pos:
            if is_cached:
                res["canonical_kp_pos"], kp_rot, kp_mir = self.get_cached(idx)
                if self.predict_lrf:
                    res["kp_rot"] = kp_rot
                    res["kp_mir"] = kp_mir
            else:
                res["canonical_kp_pos"] = kp_pos[:B].view(B, self.n_keypoints, 3)
                if self.center_canonical:
                    center = (res["canonical_kp_pos"].max(dim=1, keepdim=True).values + res["canonical_kp_pos"].min(dim=1, keepdim=True).values) / 2
                    res["canonical_kp_pos"] = res["canonical_kp_pos"] - center
                if self.predict_lrf:
                    res["kp_rot"] = kp_rot[:B].view(B, self.n_keypoints, 4)
                    res["kp_mir"] = kp_mir[:B].view(B, self.n_keypoints, 1)
                if self.use_cache:
                    self.set_cache(idx, res["canonical_kp_pos"], res.get("kp_rot", None), res.get("kp_mir", None))
            res["kp_pos"] = res["canonical_kp_pos"]
            if apply_scale and scale is not None:
                res["kp_pos"] = scale[:, None] * res["kp_pos"] 
            if apply_offset and offset is not None:
                res["kp_pos"] = res["kp_pos"] + offset[:, None]
        if return_prior:
            res["canonical_prior_kp_pos"] = kp_pos[-B*num_views:].view(B, num_views, self.n_keypoints, 3)
            if self.center_canonical:
                center = (res["canonical_prior_kp_pos"].max(dim=2, keepdim=True).values + res["canonical_prior_kp_pos"].min(dim=2, keepdim=True).values) / 2
                res["canonical_prior_kp_pos"] = res["canonical_prior_kp_pos"] - center
            res["prior_kp_pos"] = res["canonical_prior_kp_pos"]
            if apply_scale and scale is not None:
                res["prior_kp_pos"] = scale[:, None, None] * res["prior_kp_pos"]
            if apply_offset and offset is not None:
                res["prior_kp_pos"] = res["prior_kp_pos"] + offset[:, None, None]
            if self.predict_lrf:
                res["prior_kp_rot"] = kp_rot[-B*num_views:].view(B, num_views, self.n_keypoints, 4)
                res["prior_kp_mir"] = kp_mir[-B*num_views:].view(B, num_views, self.n_keypoints, 1)
        return res
