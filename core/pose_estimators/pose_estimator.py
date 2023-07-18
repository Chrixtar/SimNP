from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor
from torch.nn import Module


class PoseEstimator(Module, ABC):
    def __init__(
        self, 
        n_obj: int,
        cam_dist: float
    ) -> None:
        super(PoseEstimator, self).__init__()
        self.n_obj = n_obj
        self.cam_dist = cam_dist

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
