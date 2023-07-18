from typing import Dict, List, Optional, Tuple
import warnings

from torch import Tensor

from .extractor import Extractor
from utils.flex_embedding import FlexEmbedding


class Embedding(Extractor):
    def __init__(
        self,
        n_kp: int,
        kp_dim: int, 
        out_dim: int,
        n_obj: int,
        gpu: bool = True
    ) -> None:
        super(Embedding, self).__init__(n_kp, kp_dim, out_dim, n_obj)
        self.gpu = gpu
        self.init_emb()       # hack to keep the Embedding on cpu despite of Pytorch-Lightning moving everything to GPUs
    
    def init_emb(self):
        emb = FlexEmbedding(self.n_obj, self.n_kp * self.out_dim)
        self.emb = emb if self.gpu else [emb]

    def forward(
        self,
        idx: Tensor,
        **kwargs
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Arguments:
            idx: [B]
        Returns:
            out: [B, n_kp, out_dim]
        """
        return self.access_cpu_or_gpu_emb(self.emb, idx, self.gpu).view(-1, self.n_kp, self.out_dim), {}
    
    # Ensures saving and loading embedding despite of hack for keeping it on CPU
    def get_extra_state(self):
        emb = self.emb if self.gpu else self.emb[0]
        return {"emb": emb.get_extra_state()}
    
    def set_extra_state(self, state):
        if state is not None and "emb" in state:
            emb = self.emb if self.gpu else self.emb[0]
            emb.set_extra_state(state["emb"])
