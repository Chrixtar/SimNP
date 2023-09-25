import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def to_dict(D, dict_type=dict):
    if not isinstance(D, dict):
        return D
    D = dict_type(D)
    for k, v in D.items():
        if isinstance(v, dict):
            D[k] = to_dict(v, dict_type)
        elif isinstance(v, list):
            D[k] = [to_dict(e) for e in v]
    return D


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_nested(d: dict, *args, default=None, strict=False):
    for arg in args:
        if strict or arg in d:
            d = d[arg]
        else:
            return default
    return d

def shifted_cumsum(t, dim=0, shift=1):
    slicing = dim * (slice(None),)
    return torch.cumsum(
        torch.cat(
            (
                torch.zeros_like(t[slicing + (slice(None, shift),)]),
                t[slicing + (slice(None, -shift),)]
            ),
            dim=dim
        ),
        dim=dim
    )

def sparse_batched_mask_sampling(mask: Tensor, data: Optional[Tensor] = None):
    """
    mask: [B, ...]
    optional data: [B, ..., D]
    """
    B = mask.shape[0]
    device = mask.device
    sparse_data = torch.nonzero(mask)   # [nnz, mask.dims()]
    if data is not None:
        # [nnz, mask.dims()+D]
        sparse_data = torch.cat(
            (
                sparse_data, 
                torch.masked_select(
                    data, 
                    mask.unsqueeze(-1)
                ).view(-1, data.shape[-1])
            ), 
            dim=-1
        )
    nnz = sparse_data.shape[0]
    # Shuffle within instances
    perm = torch.randperm(nnz, device=device)
    shuffled = sparse_data[perm, :]
    idx = torch.argsort(shuffled[:, 0])
    shuffled = shuffled[idx, 1:]
    # Get extract idx
    counts = mask.int().flatten(1).sum(-1)
    min_count = counts.min()
    if min_count == 0:
        return None
    batch_start = shifted_cumsum(counts, dim=0, shift=1)
    extract_idx = (torch.arange(min_count, device=device).view(1, -1) + batch_start.view(-1, 1)).view(-1)
    res = shuffled[extract_idx].view(B, min_count, -1)
    return res

def equidistant_sphere(n: int, r: float):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    for i in range(n):
        y = 1 - (i / (n - 1 if n > 1 else 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    xyz = r * np.asarray(points)
    return xyz