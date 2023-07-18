from torch import Tensor
import torch
from torch.nn.functional import normalize

from .pose_estimator import PoseEstimator
from utils.flex_embedding import FlexEmbedding


class Embedding(PoseEstimator):
    def __init__(
        self,
        n_obj: int,
        cam_dist: float,
        gpu: bool = True
    ) -> None:
        super(Embedding, self).__init__(n_obj, cam_dist)
        self.gpu = gpu
        self.init_emb()       # hack to keep the Embedding on cpu despite of Pytorch-Lightning moving everything to GPUs
    
    def init_emb(self):
        emb = FlexEmbedding(self.n_obj, 3)
        self.emb = emb if self.gpu else [emb]

    @staticmethod
    def look_at_origin(cam_location):
        # Cam points in positive z direction
        forward = -cam_location
        forward = normalize(forward)
        tmp = torch.tensor([0., 0., -1.], device=cam_location.device)
        right = torch.cross(tmp.unsqueeze(0), forward)
        right = normalize(right)
        up = torch.cross(forward, right)
        up = normalize(up)
        mat = torch.stack((right, up, forward, cam_location), dim=-1)
        hom_vec = torch.tensor([[0., 0., 0., 1.]], device=cam_location.device)\
            .unsqueeze(0).expand(mat.shape[0], -1, -1)
        mat = torch.cat((mat, hom_vec), axis=-2)
        return mat

    def forward(
        self,
        idx: Tensor
    ) -> Tensor:
        """
        Arguments:
            idx: [B]
        Returns:
            out: [B, 4, 4]
        """
        raw_cam_params = self.access_cpu_or_gpu_emb(self.emb, idx, self.gpu)
        T = self.cam_dist * normalize(raw_cam_params)
        cam2world = self.look_at_origin(T)
        R = cam2world[:, :3, :3].transpose(-1, -2)
        T = -torch.matmul(R, cam2world[:, :3, 3:])
        extr = torch.cat((R, T), dim=-1)
        hom_vec = torch.tensor([[0., 0., 0., 1.]], device=extr.device)\
            .unsqueeze(0).expand(extr.shape[0], -1, -1)
        extr = torch.cat((extr, hom_vec), axis=-2)
        return extr

    # Ensures saving and loading embedding despite of hack for keeping it on CPU
    def get_extra_state(self):
        emb = self.emb if self.gpu else self.emb[0]
        return {"emb": emb.get_extra_state()}
    
    def set_extra_state(self, state):
        if state is not None and "emb" in state:
            emb = self.emb if self.gpu else self.emb[0]
            emb.set_extra_state(state["emb"])
