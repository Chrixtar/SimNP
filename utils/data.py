import collections
from typing import Iterable

from easydict import EasyDict as edict
import torch


def trim_tensor_dict(var, trim_size, dim=0):
    res = edict()
    for key, val in var.items():
        if isinstance(val, torch.Tensor):
            res[key] = val[dim * (slice(None),) + (slice(min(trim_size, val.shape[dim])),)]
        elif isinstance(val, dict):
            res[key] = trim_tensor_dict(val, trim_size)
        else:
            res[key] = val
    return res

def recursive_concat(batch, dim=0):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return torch.cat(batch, dim)
    elif isinstance(elem, collections.abc.Mapping):
        res = {key: recursive_concat([d[key] for d in batch]) for key in elem}
        try:
            return elem_type(res)
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return res
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(recursive_concat(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.
        try:
            return elem_type([recursive_concat(samples) for samples in transposed])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [recursive_concat(samples) for samples in transposed]

    raise TypeError(f"Unexpected type {elem_type}")


def recursive_to_cpu(var):
    var_type = type(var)
    if isinstance(var, torch.Tensor):
        res = var.detach().cpu()
        torch.cuda.empty_cache()
        return res
    elif isinstance(var, collections.abc.Mapping):
        res = {key: recursive_to_cpu(var[key]) for key in var}
        try:
            return var_type(res)
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return res
    elif isinstance(var, collections.abc.Sequence) and not isinstance(var, str):
        res = [recursive_to_cpu(e) for e in var]
        try:
            return var_type(res)
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return res
    return var

def recursive_to_gpu(var):
    var_type = type(var)
    if isinstance(var, torch.Tensor):
        res = var.cuda()
        return res
    elif isinstance(var, collections.abc.Mapping):
        res = {key: recursive_to_gpu(var[key]) for key in var}
        try:
            return var_type(res)
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return res
    elif isinstance(var, collections.abc.Sequence) and not isinstance(var, str):
        res = [recursive_to_gpu(e) for e in var]
        try:
            return var_type(res)
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return res
    return var

def split_tensor_dict(d, chunk_size, dim=0, keys: Iterable[str] = []):
    tmp = {}
    if keys:
        d_set = set(d.keys())
        key_set = set(keys)
        this_level = d_set & key_set
        deeper_level = key_set - d_set
    else:
        this_level = d.keys()
        deeper_level = []
    for k in this_level:
        if isinstance(d[k], torch.Tensor):
            tmp[k] = torch.split(d[k], chunk_size, dim)
        elif isinstance(d[k], dict):
            tmp[k] = split_tensor_dict(d[k], chunk_size, dim, deeper_level)
    num_vars = len(list(tmp.values())[0])
    res = []
    for i in range(num_vars):
        r = edict(d.copy())
        for k, v in tmp.items():
            r[k] = v[i]
        res.append(r)
    return res

def join_tensor_dicts(dl, dim=0, keys: Iterable[str] = []):
    numel = len(dl)
    if numel == 1:
        return dl[0]
    if keys:
        d_set = set(dl[0].keys())
        key_set = set(keys)
        this_level = d_set & key_set
        deeper_level = key_set - d_set
    else:
        this_level = dl[0].keys()
        deeper_level = []
    tmp = collections.defaultdict(list)
    for d in dl:
        for k in this_level:
            tmp[k].append(d[k])
    assert all(len(v) == numel for v in tmp.values())
    res = edict(dl[0].copy())
    for k, v in tmp.items():
        elem = v[0]
        if isinstance(elem, torch.Tensor):
            res[k] = torch.cat(v, dim)
        elif isinstance(elem, dict):
            res[k] = join_tensor_dicts(v, dim, deeper_level)
    return res
