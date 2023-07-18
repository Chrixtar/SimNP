from typing import Dict, List, Optional, Tuple

from pytorch3d.transforms import quaternion_apply
from torch import Tensor
import torch
from torch_knnquery import VoxelGrid

from .aggregator import Aggregator
from utils.model import define_mlp
from utils.positional_encoder import PositionalEncoder1D


class MLP(Aggregator):
    def __init__(
        self, 
        in_dim: int,
        voxel_grid: Optional[VoxelGrid], 
        k: int,
        r: float,
        max_shading_pts: int,
        ray_subsamples: int,
        out_dim: int,
        n_freqs: int, 
        layers: List[int],
        activation: str = "ReLU",
        layer_norm: bool = False,
        freq_mult: float = 1,
        detach_points: bool = True,
        norm_displacements: bool = False,
        chunk_size: int = 1000000
    ) -> None:
        super(MLP, self).__init__(in_dim, voxel_grid, k, r, max_shading_pts, ray_subsamples, out_dim, chunk_size)
        self.pos_enc = PositionalEncoder1D(n_freqs, freq_mult)
        self.detach_points = detach_points
        self.norm_displacements = norm_displacements
        self.local_field = define_mlp(layers, self.in_dim + self.pos_enc.d_out(3), self.out_dim, activation, layer_norm)

    def get_local_feat(
        self, 
        x: Tensor, 
        kp_pos: Tensor,
        kp_rot: Optional[Tensor],
        kp_mir: Optional[Tensor],
        kp_feat: Tensor,
        sample: bool,
        random_mask_local: Optional[bool] = False,
    ) -> dict:
        """
        Arguments:
            x: [B, num_tar, num_rays, num_shading_pts, 3]
            kp_pos: [B, num_kp, 3]
            kp_rot: [B, num_kp, 4]
            kp_mir: [B, num_kp, 1]
            kp_feat: [B, num_kp, in_dim]
        Return: Dictionary with
            local_feat: [num_valid_pairs, out_dim]
            shading_pts: [num_sample_pts, 3]
            mask: [B, num_tar, num_sample_rays, max_shading_pts, 1]
            ray_mask: [B, num_tar, num_rays]
            shading_idx: [num_valid_pairs]
            kp_idx: [num_valid_pairs]
            weights: [num_valid_pairs]
            num_valid_pts (int)
        """
        if self.detach_points:
            kp_pos = kp_pos.detach()
        num_kp = kp_pos.shape[1]
        # [num_valid_pts, k], [num_valid_pts, 3], [B, num_tar, num_rays, max_shading_pts, 1]
        neighbor_idx, shading_pts, mask = self.query_keypoints(x, kp_pos)
        if sample:
            # [num_sample_pts, k], [num_sample_pts, 3], [B, num_tar, num_sample_rays, max_shading_pts, 1], [B, num_tar, num_rays]
            neighbor_idx, shading_pts, mask, ray_mask = self.subsample_valid_rays(neighbor_idx, shading_pts, mask)
        else:
            ray_mask = None

        valid_mask = neighbor_idx >= 0
        neighbor_idx[~valid_mask] = 0   # for valid indices during index select
        kp = self.get_keypoint_data(neighbor_idx, valid_mask, kp_pos, kp_rot, kp_mir, kp_feat)

        num_valid_pts = neighbor_idx.shape[0]
        idx = self.mask_to_batch_ray_idx(valid_mask)                # [num_valid_pairs]
        shading_pos = shading_pts[idx, :]                           # [num_valid_pairs, 3]

        x_rel = shading_pos - kp["pos"]
        if self.norm_displacements:
            # Normalize displacement vectors by radius
            x_rel = x_rel / self.scaled_r
        weights = 1 / (torch.norm(x_rel, dim=-1) + 1.e-5)           # [num_valid_pairs]
        if "rot" in kp:
            x_rel = quaternion_apply(kp["rot"], x_rel)
        if "mir" in kp:
            x_rel = torch.stack((x_rel, self.mirror * x_rel), dim=0)# [2, num_valid_pairs, 3]
            num_mirrors = 2
        else:
            x_rel = x_rel.unsqueeze(0)
            num_mirrors = 1
        x_enc = self.pos_enc(x_rel)                                 # [num_mirrors, num_valid_pairs, d_enc]
        field_in = torch.cat(                                       # [num_mirrors, num_valid_pairs, in_dim + d_enc]
            (kp["feat"].expand(num_mirrors, -1, -1), x_enc), 
            dim=-1
        )
        field_in = field_in.flatten(0, 1)
        local_feat = torch.empty((field_in.shape[0], self.out_dim), device=field_in.device)
        chunks = torch.split(field_in, self.chunk_size, dim=0)
        # local_feat = []
        for i, chunk in enumerate(chunks):
            offset = i * self.chunk_size
            local_feat[offset:offset+chunk.shape[0]] = self.local_field(chunk)
            # local_feat.append(self.local_field(chunk))
        # local_feat = torch.cat(local_feat, dim=0)
        local_feat = local_feat.view(num_mirrors, -1, self.out_dim)
        if "mir" in kp:
            local_feat = kp["mir"] * local_feat[0] + (1-kp["mir"]) * local_feat[1]  # [num_valid_pairs, out_dim]
        else:
            local_feat = local_feat.squeeze(0)
        
        norm = torch.zeros(num_valid_pts, device=weights.device)
        norm.index_add_(0, idx, weights)
        weights = weights / norm[idx]

        kp_idx = neighbor_idx[valid_mask] % num_kp
        return {
            "local_feat": local_feat,
            "shading_pts": shading_pts,
            "mask": mask,
            "ray_mask": ray_mask,
            "shading_idx": idx,
            "kp_idx": kp_idx,
            "weights": weights,
            "num_valid_pts": num_valid_pts
        }


    def aggregate_local_feat(
        self,
        local_feat: Tensor,
        weights: Tensor,
        shading_idx: Tensor,
        num_valid_pts: int
    ) -> Tensor:
        """
        Arguments:
            local_feat: [num_valid_pairs, out_dim]
            weights: [num_valid_pairs]
            shading_idx: [num_valid_pairs]
            num_valid_pts (int)
        Return:
            feat: [num_valid_pts, out_dim]
        """
        device = local_feat.device
        weighted = weights.unsqueeze(-1) * local_feat
        feat = torch.zeros(num_valid_pts, self.out_dim, device=device)
        feat.index_add_(0, shading_idx, weighted)
        # norm = torch.zeros(num_valid_pts, device=device)
        # norm.index_add_(0, shading_idx, weights)
        # feat = feat / norm.unsqueeze(-1)
        return feat