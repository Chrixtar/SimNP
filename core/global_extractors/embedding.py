from torch import Tensor
import torch

from .global_extractor import GlobalExtractor
from utils.pinned_embedding import PinnedEmbedding


class Embedding(GlobalExtractor):
    def __init__(
        self,
        n_obj: int,
        out_dim: int,
        gpu: bool = True
    ) -> None:
        super(Embedding, self).__init__(n_obj, out_dim)
        self.emb = PinnedEmbedding(self.n_obj, self.out_dim, gpu, flex=True)

    def interpolate(self, idx: Tensor, num_steps: int) -> None:
        """
        Arguments
            idx: [n, 2]
        """
        n = len(idx)
        latents = self.emb(idx.flatten()).view(n, 2, -1)  # [n, 2, out_dim]
        step_weights = torch.linspace(0, 1, num_steps)[None, :, None]                           # [1, num_steps, 1]
        interpolated = (1 - step_weights) *  latents[:, :1] + step_weights * latents[:, 1:]     # [n, num_steps, out_dim]
        self.n_obj = n * num_steps
        self.emb.init(interpolated.flatten(0, 1))
    
    def randomize(self, idx: Tensor, num_random: int, std: float) -> None:
        """
        Arguments
            idx: [n]
        """
        n = len(idx)
        self.n_obj = n * num_random
        self.emb.init(std=std)

    def copy_selection(self, idx: Tensor, num_copies: int) -> None:
        """
        Arguments
            idx: [n]
        """
        n = len(idx)
        latents = self.emb(idx)     # [n, out_dim]
        self.n_obj = n * num_copies
        self.emb.init(latents[:, None].expand(-1, num_copies, -1).flatten(0, 1))

    def forward(
        self,
        idx: Tensor,
    ) -> Tensor:
        """
        Arguments:
            idx: [B]
        Returns:
            out: [B, out_dim]
        """
        return self.emb(idx)
