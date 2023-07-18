from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from core.fields.field import Field
from .renderer import Renderer


class NeuSRenderer(Renderer):
    def __init__(
        self,
        field: Field,
        cube_scale: float,
        depth_resolution: int,
        init_var: float,
        perturb: bool,
        anneal_end: Optional[int] = None,
        ray_limits: Optional[Tuple[float, float]] = None,
        ray_subsamples: int = 0,
        disparity_space_sampling: bool = False,
        randomize_depth_samples: bool = True,
        white_back: bool = False
    ):
        super(NeuSRenderer, self).__init__(field, cube_scale, depth_resolution, ray_limits, ray_subsamples, disparity_space_sampling, randomize_depth_samples,  white_back)
        self.register_parameter('variance', nn.Parameter(torch.tensor([init_var])))
        self.perturb = perturb
        self.anneal_end = anneal_end


    def get_alpha(
        self,  
        sdf: Tensor, 
        depths: Tensor,
        rays_d: Tensor, 
        gradients: Tensor,
        cos_anneal_ratio: float = 0.0,
    ):
        """
        Arguments:
            sdf: [B, num_tar, num_sample_rays, max_shading_pts, 1]
            depths: [B, num_tar, num_sample_rays, max_shading_pts, 1]
            rays_d: [B, num_tar, num_sample_rays, 1, 3]
            gradients: [B, num_tar, num_sample_rays, max_shading_pts, 3]
        Returns:
            alpha: [B, num_tar, num_sample_rays, max_shading_pts, 1]
        """

        deltas = torch.cat((depths[..., 1:, :] - depths[..., :-1, :], \
            torch.zeros_like(depths[..., :1, :])), dim=-2)  # [B, num_tar, num_sample_rays, max_shading_pts, 1]

        inv_s = torch.exp(self.variance * 10.0).clip(1e-6, 1e6)
        true_cos = (rays_d * gradients).sum(-1, keepdim=True) # [B, num_tar, num_sample_rays, max_shading_pts, 1]

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * deltas * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
        return alpha

    def render(
        self, 
        kp_pos: Tensor, 
        kp_rot: Optional[Tensor],
        kp_mir: Optional[Tensor],
        kp_feat: Tensor,
        global_feat: Optional[Tensor],
        rays_o: Tensor, 
        rays_d: Tensor, 
        ray_start: Tensor, 
        ray_end: Tensor, 
        sample: bool,
        global_step: int = 0,
        return_channels: bool = True,
        return_kp_weights: bool = False
    ):
        """
        Returns: Dictionary with
            mask: [B, num_tar, num_sample_rays, 1]
            depth: [B, num_tar, num_sample_rays, 1]
            ray_sample_mask: [B, num_tar, num_rays]
            grad: [B, num_tar, num_sample_rays, max_shading_pts, 3]
            valid_pts_mask: [B, num_tar, num_sample_rays, max_shading_pts, 1]
            optional channels: [B, num_tar, num_sample_rays, 3]
            optional kp_weights: [B, num_tar, num_sample_rays, num_kp]
                with num_kp = max {idx[kp] for kp with at least one weight > 0}
        """
        B, num_tar, num_rays = rays_o.shape[:3]
        depths = self.sample(ray_start.flatten(0, 1), ray_end.flatten(0, 1))         # [B*num_tar, num_rays, samples_per_ray, 1]
        depths = depths.view(B, num_tar, num_rays, -1, 1)

        if self.perturb:
            t_rand = torch.rand_like(depths) - 0.5
            depths = depths + t_rand * 2.0 / self.depth_resolution

        # Add sample dimension
        rays_o = rays_o.unsqueeze(-2)
        rays_d = rays_d.unsqueeze(-2)

        sample_coordinates = rays_o + depths * rays_d  # [B, num_tar, num_rays, samples_per_ray, 3]
        with torch.inference_mode(False):
            sample_coordinates.requires_grad_(True)
            out = self.field(
                sample_coordinates, 
                rays_d, 
                kp_pos,
                kp_rot,
                kp_mir, 
                kp_feat, 
                global_feat,
                sample=sample, 
                return_channels=return_channels, 
                return_mask=True, 
                return_pts=True, 
                return_grad=True,
                return_kp_weights=return_kp_weights
            )
        # [B, num_tar, num_sample_rays, max_shading_pts, 1]
        depths = self.get_depths_from_shading_pts(out["pts"], out["mask"], out["ray_sample_mask"], rays_o, rays_d, ray_end)
        sampled_rays_d = torch.masked_select(rays_d, out["ray_sample_mask"][..., None, None]).view(B, num_tar, -1, 1, 3) if out["ray_sample_mask"] is not None else rays_d

        cos_anneal_ratio = min(1.0, global_step / self.anneal_end) if self.anneal_end is not None else 1.0
        res = self.ray_march(
            out["shape"], 
            depths, 
            out.get("channels", None),
            out.get("kp_weights", None),
            out.get("mask", None),
            rays_d=sampled_rays_d, 
            gradients=out["grad"],
            cos_anneal_ratio=cos_anneal_ratio
        )
        res["grad"] = out["grad"]
        res["ray_sample_mask"] = out["ray_sample_mask"]
        res["valid_pts_mask"] = out["mask"]
        return res
