from typing import Optional

import torch
from torch import Tensor


def get_vis_intr(intr, vis_size, img_size):
    scale_factor = vis_size / img_size
    center = 0.5 * (scale_factor - 1)
    scale_transform = torch.tensor([[scale_factor, 0, center], [0, scale_factor, center], [0, 0, 1]], device=intr.device)
    scaled_intr = torch.matmul(scale_transform, intr)
    return scaled_intr


def img_to_cam(v: Tensor, intr: Tensor, depth: Optional[Tensor] = None):
    """
    Arguments:
        v: [..., num_points, 2]
        intr: [..., 3, 3]
        depth: [..., num_points]
    Returns:
        v_cam: [..., num_points, 3]
    """
    fx = intr[..., 0, 0]
    fy = intr[..., 1, 1]
    cx = intr[..., 0, 2]
    cy = intr[..., 1, 2]
    sk = intr[..., 0, 1]

    x_cam = v[..., 0]
    y_cam = v[..., 1]
    if depth is None:
        depth = torch.ones(v.shape[:-1], device=intr.device)

    x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y_cam/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * depth
    y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * depth

    v_cam = torch.stack((x_lift, y_lift, depth), dim=-1)
    return v_cam

def cam_to_world(v: Tensor, extr: Tensor):
    """
    Arguments:
        v: [..., num_points, 3]
        extr: [..., 4, 4]   (world2cam)
    """
    # [..., 3, 3], [..., 3]
    R, T = extr[..., :3, :3], extr[..., :3, 3]
    # Apply inverse (negative) translation
    v = v - T.unsqueeze(-2)
    # Apply inverse rotation (right matmul = left matmul with transpose = inverse)
    v = v @ R
    return v

def img_to_world(v: Tensor, intr: Tensor, extr: Tensor, depth: Optional[Tensor] = None):
    """
    Arguments:
        v: [..., num_points, 2]
        intr: [..., 3, 3],
        extr: [..., 4, 4]
        depth: [..., num_points]
    Returns:
        v_world: [..., num_points, 3]
    """
    v_cam = img_to_cam(v, intr, depth)
    v_world = cam_to_world(v_cam, extr)
    return v_world

def to_cam(x, extr):
    x_hom = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
    x_cam = (x_hom @ extr.transpose(-1, -2))[..., :3]
    return x_cam

def get_keypoint_projections(kp_pos, extr, intr, pix_scale=None, pix_offset=None, eps=1.e-5):
    kp_cam = to_cam(kp_pos, extr)
    depth = torch.abs(kp_cam[..., 2])
    min_depth = depth.min(dim=-1, keepdim=True).values
    max_depth = depth.max(dim=-1, keepdim=True).values
    depth = (depth - min_depth) / (max_depth - min_depth + eps)

    kp_img = torch.matmul(kp_cam, intr.transpose(-2, -1))
    kp_pix = kp_img[..., :-1] / kp_img[..., -1:]    # [B, kp_num, 2]
    if pix_offset is not None:
        kp_pix = kp_pix - pix_offset.unsqueeze(-2)
    if pix_scale is not None:
        kp_pix = kp_pix / pix_scale.unsqueeze(-2)
    return kp_pix, depth

def get_cam_loc_in_world(extr: Tensor) -> Tensor:
    """
    Arguments:
        extr: [..., 4, 4]
    Returns:
        cam_loc_in_world: [..., 3]
    """
    R = extr[..., :3, :3].transpose(-1, -2)
    cam_loc_in_world = - torch.matmul(R, extr[..., :3, 3:])
    return cam_loc_in_world[..., 0]

def mirror_world_to_cam(extr: Tensor) -> Tensor:
    extr_mir = extr.clone()
    extr_mir[..., :3, :3] *= torch.tensor([-1, 1, -1], device=extr_mir.device).view(-1, 1)
    extr_mir[..., :3, 3] *= torch.tensor([1, -1, 1], device=extr_mir.device)
    return extr_mir