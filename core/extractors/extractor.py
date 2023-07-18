from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from torch import Tensor
from torch.nn import Module


class Extractor(Module, ABC):
    def __init__(self, n_kp: int, kp_dim: int, out_dim: int, n_obj: int) -> None:
        super(Extractor, self).__init__()
        self.n_kp = n_kp
        self.kp_dim = kp_dim
        self.out_dim = out_dim
        self.n_obj = n_obj

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
        idx: Optional[Tensor] = None,
        kp_pos: Optional[Tensor] = None,
        kp_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Arguments:
            idx: [B]
            kp_pos: [B, num_kp, 3]
            kp_ids: [num_kp, kp_dim]
        Returns:
            extracted_feat: [B, num_kp, out_dim]
            optional dict inter_res
        """
        raise NotImplementedError()
