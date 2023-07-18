from typing import Optional

import torch
from torch import Tensor


def get_cam_points(pix: Optional[Tensor], intr: Tensor, resolution: int) -> Tensor:
    B, M = intr.shape[0], resolution ** 2

    fx = intr[:, 0, 0]
    fy = intr[:, 1, 1]
    cx = intr[:, 0, 2]
    cy = intr[:, 1, 2]
    sk = intr[:, 0, 1]

    if pix is None:
        u = torch.arange(resolution, dtype=torch.float32, device=intr.device).add_(0.5)
        uv = torch.stack(torch.meshgrid(u, u, indexing='ij')) # * (1./resolution)
        uv = uv.flip(0).reshape(2, -1).transpose(1, 0)  # [resolution**2, 2]
        pix = uv.expand(B, -1, -1)
    else:
        pix = pix.flatten(-2, -1).transpose(-1, -2)

    x_cam = pix[:, :, 0].view(B, -1)
    y_cam = pix[:, :, 1].view(B, -1)
    z_cam = torch.ones((B, M), device=intr.device)

    x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y_cam/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
    y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

    cam_rel_points = torch.stack((x_lift, y_lift, z_cam), dim=-1)
    return cam_rel_points

def get_rays_from_cam_points(extr: Tensor, cam_points: Tensor) -> Tensor:
    """
    cam2world = extr.clone()
    # watch out side effects --> use extr still
    cam2world[:, :3, :3] = extr[:, :3, :3].transpose(-1, -2)
    cam2world[:, :3, 3:] = - torch.matmul(cam2world[:, :3, :3], extr[:, :3, 3:])
    """
    R = extr[:, :3, :3].transpose(-1, -2)
    T = -torch.matmul(R, extr[:, :3, 3:])
    cam2world = torch.cat((R, T), dim=-1)
    hom_vec = torch.tensor([[0., 0., 0., 1.]], device=extr.device)\
        .unsqueeze(0).expand(extr.shape[0], -1, -1)
    cam2world = torch.cat((cam2world, hom_vec), axis=-2)
    cam_locs_world = cam2world[:, :3, 3].contiguous()

    cam_rel_points = torch.cat((cam_points, torch.ones_like(cam_points[..., :1])), dim=-1)
    world_rel_points = torch.bmm(cam2world, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3].contiguous()

    ray_dirs = world_rel_points - cam_locs_world[:, None, :]
    ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

    ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

    return ray_origins, ray_dirs

def get_rays(pix: Optional[Tensor], extr: Tensor, intr: Tensor, resolution: int):
    """
    Create batches of rays and return origins and directions.

    pix: [B, 2, resolution, resolution]
    extr: (N, 4, 4)
    intr: (N, 3, 3)
    resolution: int

    ray_origins: (N, M, 3)
    ray_dirs: (N, M, 3)
    """
    cam_points = get_cam_points(pix, intr, resolution)
    return get_rays_from_cam_points(extr, cam_points)
