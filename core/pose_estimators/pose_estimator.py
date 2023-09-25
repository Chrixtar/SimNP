from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor
from torch.nn import Module


class PoseEstimator(Module, ABC):
    def __init__(
        self, 
        n_obj: int,
        cam_dist: float,
        n_repeats: int = 1
    ) -> None:
        super(PoseEstimator, self).__init__()
        assert n_obj % n_repeats == 0, "n_repeats has to divide n_obj"
        self.n_obj = n_obj
        self.cam_dist = cam_dist
        self.n_repeats = n_repeats

    @abstractmethod
    def forward(
        self,
        idx: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Arguments:
            idx: [B]
        Returns:
            extr: [B, 4, 4]
        """
        raise NotImplementedError()
