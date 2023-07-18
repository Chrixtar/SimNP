from typing import Dict, Tuple

from torch import Tensor

from .extractor import Extractor
from utils.pinned_embedding import PinnedEmbedding


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
        self.emb = PinnedEmbedding(self.n_obj, self.n_kp * self.out_dim, gpu, flex=True)

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
        return self.emb(idx).view(-1, self.n_kp, self.out_dim), {}
