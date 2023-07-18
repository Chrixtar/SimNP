from typing import Optional

from torch import Tensor
import torch
from torch.nn import init

from .global_extractor import GlobalExtractor
from utils.flex_embedding import FlexEmbedding


class Embedding(GlobalExtractor):
    def __init__(
        self,
        n_obj: int,
        out_dim: int,
        gpu: bool = True
    ) -> None:
        super(Embedding, self).__init__(n_obj, out_dim)
        self.gpu = gpu
        self.init_emb()       # hack to keep the Embedding on cpu despite of Pytorch-Lightning moving everything to GPUs
    
    def init_emb(self, std: Optional[float]):
        emb = FlexEmbedding(self.n_obj, self.out_dim)
        self.emb = emb if self.gpu else [emb]
        if std is not None:
            init.normal_(emb.weight, std=std)

    def interpolate(self, idx: Tensor, num_steps: int) -> None:
        """
        Arguments
            idx: [n, 2]
        """
        n = len(idx)
        latents = self.access_cpu_or_gpu_emb(self.emb, idx.flatten(), self.gpu).view(n, 2, -1)  # [n, 2, out_dim]
        step_weights = torch.linspace(0, 1, num_steps)[None, :, None]                           # [1, num_steps, 1]
        interpolated = (1 - step_weights) *  latents[:, :1] + step_weights * latents[:, 1:]     # [n, num_steps, out_dim]
        self.n_obj = n * num_steps
        self.init_emb(interpolated.flatten(0, 1))
    
    def randomize(self, idx: Tensor, num_random: int, std: float) -> None:
        """
        Arguments
            idx: [n]
        """
        n = len(idx)
        self.n_obj = n * num_random
        self.init_emb(std)

    def copy_selection(self, idx: Tensor, num_copies: int) -> None:
        """
        Arguments
            idx: [n]
        """
        n = len(idx)
        latents = self.access_cpu_or_gpu_emb(self.emb, idx, self.gpu)     # [n, out_dim]
        self.n_obj = n * num_copies
        self.init_emb(latents[:, None].expand(-1, num_copies, -1).flatten(0, 1))

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
        return self.access_cpu_or_gpu_emb(self.emb, idx, self.gpu)
    
    # Ensures saving and loading embedding despite of hack for keeping it on CPU
    def get_extra_state(self):
        emb = self.emb if self.gpu else self.emb[0]
        return {"emb": emb.get_extra_state()}
    
    def set_extra_state(self, state):
        if state is not None and "emb" in state:
            emb = self.emb if self.gpu else self.emb[0]
            emb.set_extra_state(state["emb"])
