from abc import ABC, abstractmethod
from typing import List, Union

from torch import Tensor
from torch.nn import Module

from utils.flex_embedding import FlexEmbedding


class GlobalExtractor(Module, ABC):
    def __init__(self, n_obj: int, out_dim: int) -> None:
        super(GlobalExtractor, self).__init__()
        self.n_obj = n_obj
        self.out_dim = out_dim
    
    def interpolate(self, idx: Tensor, num_steps: int) -> None:
        """
        Arguments
            idx: [n, 2]
        """
        raise NotImplementedError()

    def randomize(self, idx: Tensor, num_random: int, std: float) -> None:
        """
        Arguments
            idx: [n]
        """
        raise NotImplementedError()

    def copy_selection(self, idx: Tensor, num_copies: int) -> None:
        """
        Arguments
            idx: [n]
        """
        raise NotImplementedError()

    @staticmethod
    def access_cpu_or_gpu_emb(emb: Union[FlexEmbedding, List[FlexEmbedding]], idx: Tensor, gpu: bool):
        device = idx.device
        if not gpu:
            emb = emb[0]
            idx = idx.cpu()
        return emb(idx).to(device=device)

    @abstractmethod
    def forward(
        self,
        idx: Tensor,
    ) -> Tensor:
        """
        Arguments:
            idx: [B]
        Returns:
            global_feat: [B, out_dim]
        """
        raise NotImplementedError()
