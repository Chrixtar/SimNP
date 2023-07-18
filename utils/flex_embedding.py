import warnings
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Embedding


class FlexEmbedding(Embedding):

    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None, 
        norm_type: float = 2., 
        scale_grad_by_freq: bool = False,
        sparse: bool = False, 
        _weight: Optional[Tensor] = None,
        device=None, 
        dtype=None,
        save_mean: bool = False,
        init_with_mean: bool = False
    ) -> None:
        super(FlexEmbedding, self).__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            device,
            dtype
        )
        self.save_mean = save_mean
        self.init_with_mean = init_with_mean
    
    def get_extra_state(self):
        res = {"weight": self.weight}
        if self.save_mean:
            res["mean"] = self.weight.detach().mean(dim=0)
        return res
    
    def set_extra_state(self, state):
        if state is not None:
            if "weight" in state and self.weight.shape == state["weight"].shape:
                with torch.no_grad():
                    self.weight.copy_(state["weight"])
            else:
                warnings.warn(f"Found unequal shapes of embeddings in module and state_dict. \
                                Continue with re-initialized embedding.")
                if self.init_with_mean:
                    if "mean" in state:
                        mean = state["mean"]
                    elif "weight" in state:
                        mean = state["weight"].mean(dim=0)
                    else:
                        warnings.warn(f"Tried to initialize with mean, but neither mean nor weight in state_dict. \
                                      Continue with random initialization.")
                        return
                    mean_dim = mean.shape[0]
                    assert mean_dim == self.embedding_dim, \
                        f"Tried to initialize with mean, but dimension of mean {mean_dim} does not match embedding size {self.embedding_dim}"
                    with torch.no_grad():
                        self.weight.copy_(mean.repeat(self.num_embeddings, 1))

    def state_dict(self, *args, **kwargs):
        if args:
            return args[0]
        return kwargs["destination"]
    
    def _load_from_state_dict(self, *args, **kwargs):
        return