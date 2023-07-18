from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from torch import Tensor
from torch.nn import Module

from utils.flex_embedding import FlexEmbedding


class PoseEstimator(Module, ABC):
    def __init__(
        self, 
        n_obj: int,
        cam_dist: float
    ) -> None:
        super(PoseEstimator, self).__init__()
        self.n_obj = n_obj
        self.cam_dist = cam_dist
    
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
        idx: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Arguments:
            idx: [B]
        Returns:
            extr: [B, 4, 4]
        """
        raise NotImplementedError()
