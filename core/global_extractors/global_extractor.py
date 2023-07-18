from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module


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
