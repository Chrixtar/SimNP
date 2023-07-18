from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch_knnquery import VoxelGrid

from utils.model import define_mlp
from utils.positional_encoder import PositionalEncoder1D

from . import aggregators


class Field(Module, ABC):
    def __init__(
        self, 
        in_dim: int,
        global_dim: Optional[int],
        voxel_grid: Optional[VoxelGrid], 
        aggregator: dict,
        global_layers: Optional[List[int]] = None,
        global_n_freqs: Optional[int] = None,
        global_freq_mult: Optional[float] = 1,
        nerf: bool = True, 
        use_dir: bool = True,
        aggregate_shape: bool = False,
        activation: str = "ReLU",
        layer_norm: bool = False,
        random_mask_local: bool = False
    ) -> None:
        super(Field, self).__init__()
        self.global_dim = global_dim
        self.aggregator: aggregators.Aggregator = getattr(aggregators, aggregator.network)(in_dim, voxel_grid, **aggregator.kwargs)
        self.hid_dim = self.aggregator.out_dim
        assert (self.global_dim is None) == (global_layers is None), "Defined global embedding but no global field or vice versa"
        if global_layers is not None:
            global_in_dim = self.global_dim
            if global_n_freqs is not None:
                self.global_pos_enc = PositionalEncoder1D(global_n_freqs, global_freq_mult)
                global_in_dim += self.global_pos_enc.d_out(3)
            else:
                self.global_pos_enc = None
                global_in_dim += 3
            self.global_field = define_mlp(global_layers, global_in_dim, self.hid_dim, activation, layer_norm)
        else:
            self.global_field = None
        self.nerf = nerf
        self.use_dir = use_dir
        self.aggregate_shape = aggregate_shape
        self.random_mask_local = random_mask_local
        if self.nerf:
            self.shape_act = lambda x: F.softplus(x - 1)   # activation bias of -1 makes things initialize better
        else:
            self.shape_act = lambda x: x
    
    def get_global_feat(
        self,
        x: Tensor,
        mask: Tensor,
        global_feat: Tensor,
        offset: Tensor,
        scale: Tensor,
    ) -> Tensor:
        """
        Arguments:
            x: [num_sample_pts, 3]
            mask: [B, num_tar, num_sample_rays, max_shading_pts, 1]
            global_feat: [B, global_dim]
            offset: [B, 3]
            scale: [B, 3]
        Return:
            feat: [num_sample_pts, out_dim]
        """
        global_input = torch.masked_select(global_feat[:, None, None, None], mask).view(-1, self.global_dim)    # [num_sample_pts, global_dim]
        if offset is not None:
            print(offset.size())
            offset =  torch.masked_select(offset[:, None, None, None], mask).view(-1, 3)
            x = x-offset
        if self.global_pos_enc is not None:
            x = self.global_pos_enc(x)
        global_input = torch.cat((global_input, x), dim=1)
        feat = self.global_field(global_input)
        return feat

    @abstractmethod
    def get_shape(
        self, 
        feat: Tensor,
    ) -> Tensor:
        """
        Arguments:
            feat: [num_valid_pts, hid_dim]
        Return:
            shape: [num_valid_pts, 1]
        """
        raise NotImplementedError()

    @abstractmethod
    def get_channels(
        self,
        feat: Tensor, 
        ray_dir: Optional[Tensor],
    ) -> Tensor:
        """
        Arguments:
            feat: [num_valid_pts, hid_dim]
            ray_dir: [num_valid_pts, 3]
        Return:
            channels: [num_valid_pts, out_dim]
        """
        raise NotImplementedError()

    def scatter_shape(self, mask: Tensor, shape: Tensor) -> Tensor:
        if self.nerf:
            res = torch.zeros_like(mask, dtype=torch.float)
        else:
            res = torch.full_like(mask, fill_value=self.aggregator.scaled_r, dtype=torch.float)
        res.masked_scatter_(mask, shape)
        return res

    def scatter(self, mask: Tensor, source: Tensor) -> Tensor:
        val_dim = source.shape[-1]
        out_shape = mask.shape[:-1] + (val_dim,)
        res = torch.zeros(out_shape, device=source.device)
        res.masked_scatter_(mask.expand(*out_shape), source)
        return res

    def forward(
        self, 
        x: Tensor,
        ray_dir: Optional[Tensor],
        kp_pos: Tensor,
        kp_rot: Optional[Tensor],
        kp_mir: Optional[Tensor],
        kp_feat: Tensor,
        global_feat: Optional[Tensor],
        offset: Optional[Tensor],
        scale: Optional[Tensor],
        sample: bool,
        return_channels: bool = True,
        return_mask: bool = False,
        return_pts: bool = False,
        return_grad: bool = False,
        return_kp_weights: bool = False
    ) -> Dict[str, Tensor]:
        """
        Arguments:
            x: [B, num_tar, num_rays, num_shading_pts, 3]
            ray_dir: [B, num_tar, num_rays, 1, 3]
            kp_pos: [B, num_kp, 3]
            kp_rot: [B, num_kp, 4]
            kp_mir: [B, num_kp, 1]
            kp_feat: [B, num_kp, kp_dim]
            global_feat: [B, global_dim]
            offset: [B, 3],
            scale: [B, 3],
        Return:
            Dictionary with:
            shape: [B, num_tar, num_sample_rays, max_shading_pts, 1]
            ray_sample_mask: [B, num_tar, num_rays]
            optional channels: [num_valid_pts, 3]
            optional mask: [B, num_tar, num_sample_rays, max_shading_pts, 1]
            optional pts: [B, num_tar, num_sample_rays, max_shading_pts, 3]
            optional grad: [B, num_tar, num_sample_rays, max_shading_pts, 3]
            optional kp_weights: Dictionary with
                shading_idx: [num_valid_pairs]
                kp_idx: [num_valid_pairs]
                weights: [num_valid_pairs]
        """
        B, num_tar = x.shape[:2]
        agg_res = self.aggregator.get_local_feat(x, kp_pos, kp_rot, kp_mir, kp_feat, sample, self.random_mask_local)
        agg_kwargs = {k: agg_res[k] for k in ("weights", "shading_idx", "num_valid_pts")}
        if global_feat is not None:
            global_sample_feat = self.get_global_feat(agg_res["shading_pts"], agg_res["mask"], global_feat, offset, scale)  # [num_valid_pts, hid_dim]
        else:
            global_sample_feat = None
        if self.aggregate_shape:
            if global_sample_feat is not None:
                agg_res["local_feat"] = agg_res["local_feat"] + global_sample_feat[agg_res["shading_idx"], :]
            shape = self.get_shape(agg_res["local_feat"]) # [num_valid_pairs, 1]
            shape = self.shape_act(shape)
            if return_channels:
                local_feat = torch.cat((shape, agg_res["local_feat"]), dim=-1)    # [num_valid_pairs, hid_dim+1]
                # [num_valid_pts, hid_dim+1]
                feat = self.aggregator.aggregate_local_feat(local_feat, **agg_kwargs)
                shape = feat[:, :1]
                feat = feat[:, 1:]
            else:
                # [num_valid_pts, 1]
                shape = self.aggregator.aggregate_local_feat(shape, **agg_kwargs)
        else:
            # [num_valid_pts, hid_dim]
            feat = self.aggregator.aggregate_local_feat(agg_res["local_feat"], **agg_kwargs)
            if global_sample_feat is not None:
                if self.random_mask_local:
                    num_rays = agg_res["mask"].shape[2]
                    ray_mask = (torch.rand((B, num_tar, num_rays), device=feat.device) > 0.5).float()
                    ray_mask = torch.masked_select(ray_mask[..., None, None], agg_res["mask"]).unsqueeze(-1) # [num_valid_pts, 1]
                    feat = feat * ray_mask
                feat = feat + global_sample_feat
            shape = self.get_shape(feat)
            shape = self.shape_act(shape)
        
        res = {
            "shape": self.scatter_shape(agg_res["mask"], shape), 
            "ray_sample_mask": agg_res["ray_mask"],
        }
        if return_channels:
            if ray_dir is not None:
                if sample:
                    ray_dir = torch.masked_select(ray_dir, res["ray_sample_mask"][..., None, None]).view(B, num_tar, -1, 1, 3)
                ray_dir = torch.masked_select(ray_dir, agg_res["mask"]).view(-1, 3)  # [num_sample_pts, 3]
            channels = self.get_channels(feat, ray_dir)
            channels = torch.sigmoid(channels)
            res["channels"] = channels
        if return_pts:
            res["pts"] = self.scatter(agg_res["mask"], agg_res["shading_pts"])
        if return_grad:
            d_output = torch.ones_like(shape, requires_grad=False)
            grad = torch.autograd.grad(outputs=shape, inputs=agg_res["shading_pts"], grad_outputs=d_output, create_graph=True, retain_graph=True)[0]
            res["grad"] = self.scatter(agg_res["mask"], grad)
        if return_kp_weights:
            res["kp_weights"] = {k: agg_res[k] for k in ("shading_idx", "kp_idx", "weights")}
        if return_mask:
            res["mask"] = agg_res["mask"]
        return res
