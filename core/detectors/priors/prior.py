from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn, Tensor

from utils.positional_encoder import PositionalEncoder1D
from core.renderers.ray_sampling import get_rays


class Prior(nn.Module, ABC):

    def __init__(
        self,
        out_dim: int,
        segmentation_mask: bool = False,
        pix_encoding: bool = False,
        ray_encoding: bool = False,
        pix_encoder: Optional[dict] = None,
        pos_encoder: Optional[dict] = None,
        dir_encoder: Optional[dict] = None,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.segmentation_mask = segmentation_mask
        self.pix_encoding= pix_encoding
        self.ray_encoding = ray_encoding
        self.in_dim = 3
        if self.segmentation_mask:
            self.in_dim += 1
        if self.pix_encoding:
            if pix_encoder is not None:
                self.pix_encoder = PositionalEncoder1D(**pix_encoder)
                self.in_dim += self.pix_encoder.d_out(2)
            else:
                self.pix_encoder = None
                self.in_dim += 2
        if self.ray_encoding:
            if pos_encoder is not None:
                self.pos_encoder = PositionalEncoder1D(**pos_encoder)
                self.in_dim += self.pos_encoder.d_out(3)
            else:
                self.pos_encoder = None
                self.in_dim += 3
            if dir_encoder is not None:
                self.dir_encoder = PositionalEncoder1D(**dir_encoder)
                self.in_dim += self.dir_encoder.d_out(3)
            else:
                self.dir_encoder = None
                self.in_dim += 3

    def get_pix_encoding(
        self,
        img: Tensor,
        pix: Optional[Tensor]
    ) -> Tensor:
        """
        Arguments:
            img: [B, 3, resolution, resolution]
            pix: [B, 2, resolution, resolution]
        Returns:
            out: [B, 3+pix_enc_dim, resolution, resolution]
        """
        B, _, resolution = img.shape[:3]
        if pix is None:
            u = torch.linspace(0, 1, resolution, dtype=torch.float32, device=img.device) - 0.5
            pix = torch.stack(torch.meshgrid(u, u, indexing='ij')).flip(0).unsqueeze(0)   # [2, resolution, resolution]
        if self.pix_encoder is not None:
            pix = self.pix_encoder(pix.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = torch.cat((img, pix.expand(B, -1, -1, -1)), dim=1)
        return out

    def get_ray_encoding(
        self,
        img: Tensor,
        pix: Optional[Tensor],
        extr: Tensor, 
        intr: Tensor
    ) -> Tensor:
        """
        Arguments:
            img: [B, 3, resolution, resolution]
            pix: [B, 2, resolution, resolution]
            extr: [B, 4, 4]
            intr: [B, 3, 3]
        Returns:
            out: [B, in_dim, resolution, resolution]
        """
        B, _, resolution = img.shape[:3]
        rays_o, rays_d = get_rays(pix, extr, intr, resolution)
        rays_o = rays_o.view(B, resolution, resolution, -1)
        rays_d = rays_d.view(B, resolution, resolution, -1)
        if self.pos_encoder is not None:
            rays_o = self.pos_encoder(rays_o)
        if self.dir_encoder is not None:
            rays_d = self.dir_encoder(rays_d)
        out = torch.cat((img, rays_o.permute(0, 3, 1, 2), rays_d.permute(0, 3, 1, 2)), dim=1)
        return out

    @abstractmethod
    def forward(
        self, 
        img: Tensor,
        mask: Optional[Tensor],
        pix: Optional[Tensor],
        extr: Optional[Tensor], 
        intr: Optional[Tensor]
    ) -> Tensor:
        raise NotImplementedError()
