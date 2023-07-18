from typing import Optional

import torch
from torch import nn, Tensor
import torchvision

from .prior import Prior


class ResNet(Prior):

    def __init__(
        self,
        out_dim: int,
        segmentation_mask: bool = False,
        pix_encoding: bool = False,
        ray_encoding: bool = False,
        pix_encoder: Optional[dict] = None,
        pos_encoder: Optional[dict] = None,
        dir_encoder: Optional[dict] = None,
        network: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.
    ):
        super().__init__(
            out_dim, 
            segmentation_mask,
            pix_encoding,
            ray_encoding,
            pix_encoder, 
            pos_encoder, 
            dir_encoder
        )
        self.model = getattr(torchvision.models, network)(pretrained=pretrained)
        if self.in_dim != 3:
            self.model.conv1 = nn.Conv2d(
                self.in_dim,
                self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=False
            )
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.model.fc.in_features, self.out_dim)
        )
    
    def forward(
        self, 
        img: Tensor,
        mask: Optional[Tensor],
        pix: Optional[Tensor],
        extr: Optional[Tensor], 
        intr: Optional[Tensor]
    ) -> Tensor:
        if self.segmentation_mask:
            img = torch.cat((img, mask), dim=1)
        if self.pix_encoding:
            img = self.get_pix_encoding(img, pix)
        if self.ray_encoding:
            img = self.get_ray_encoding(img, pix, extr, intr)   
        return self.model(img)
