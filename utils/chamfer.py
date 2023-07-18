# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds


def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: str
) -> None:
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != X.shape[0]:
                raise ValueError("Expected lengths to be of shape (N,)")
            if lengths.max() > X.shape[1]:
                raise ValueError("A length value was too long")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
    norm: int = 2,
    return_xy: bool = True,
    return_yx: bool = True
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.

    Returns:
        dictionary containing
        if return_xy:
            - cham_xy: Tensor giving the reduced distance from pointclouds in x to pointclouds in y
            if x_normals and y_normals are not None:
                - cham_norm_xy: Reduced cosine distance of normals from pointclouds in x to pointclouds in y
        if return_yx:
            - cham_yx: Tensor giving the reduced distance from pointclouds in y to pointclouds in x
            if x_normals and y_normals are not None:
                - cham_norm_yx: Reduced cosine distance of normals from pointclouds in y to pointclouds in x
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm = x.new_zeros(())
    # cham_norm_x = x.new_zeros(())
    # cham_norm_y = x.new_zeros(())
    cham = []
    lengths = []
    if return_xy:
        x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
        cham_x = x_nn.dists[..., 0]     # (N, P1)
        if is_x_heterogeneous:
            cham_x[x_mask] = 0.0
        cham_x = cham_x.sum(dim=1)      # (N,)
        cham.append(cham_x)
        lengths.append(x_lengths)
    if return_yx:
        y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, norm=norm, K=1)
        cham_y = y_nn.dists[..., 0]     # (N, P2)
        if is_y_heterogeneous:
            cham_y[y_mask] = 0.0
        cham_y = cham_y.sum(dim=1)      # (N,)
        cham.append(cham_y)
        lengths.append(y_lengths)
    cham = torch.stack(cham, dim=0)         # (1 or 2, N)
    lengths = torch.stack(lengths, dim=0)   # (1 or 2, N)

    if weights is not None:
        cham *= weights.view(1, N)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        cham_norm = []
        if return_xy:
            x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
            cham_norm_x = 1 - torch.abs(
                F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
            )
            if is_x_heterogeneous:
                cham_norm_x[x_mask] = 0.0
            cham_norm_x = cham_norm_x.sum(dim=1)
            cham_norm.append(cham_norm_x)
        if return_yx:
            y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]
            cham_norm_y = 1 - torch.abs(
                F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
            )
            if is_y_heterogeneous:
                cham_norm_y[y_mask] = 0.0
            cham_norm_y = cham_norm_y.sum(dim=1)
            cham_norm.append(cham_norm_y)
        cham_norm = torch.stack(cham_norm, dim=0)   # (1 or 2, N)
        if weights is not None:
            cham_norm *= weights.view(1, N)

    # Apply point reduction
    if point_reduction == "mean":
        lengths_clamped = lengths.clamp(min=0)
        cham /= lengths_clamped
        if return_normals:
            cham_norm /= lengths_clamped

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham = cham.sum(dim=1)  # (1 or 2,)
        if return_normals:
            cham_norm = cham_norm.sum(dim=1)    # (1 or 2,)
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else max(N, 1)
            cham /= div
            if return_normals:
                cham_norm /= div

    res = {}
    if return_xy:
        res["cham_xy"] = cham[0]
        if return_normals:
            res["cham_norm_xy"] = cham_norm[0]
    if return_yx:
        res["cham_yx"] = cham[-1]
        if return_normals:
            res["cham_norm_yx"] = cham_norm[-1]

    return res