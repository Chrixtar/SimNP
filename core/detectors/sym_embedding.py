from typing import Dict, List, Optional, Tuple
import warnings

from easydict import EasyDict as edict
import torch
from torch import Tensor
from torch.nn import Parameter, init

from .detector import Detector
from utils.model import define_mlp
from utils.flex_embedding import FlexEmbedding


class SymEmbedding(Detector):
    def __init__(
        self,
        in_dim: int,
        cube_scale: float,
        n_obj: int,
        n_keypoints: int,
        decoder: edict,
        emb_dim: int,
        gpu: bool = True,
        predict_lrf: bool = False,
        pos_head: Optional[edict] = None,
        lrf_head: Optional[edict] = None,
        prior: Optional[edict] = None,
        scale_emb: bool = False,
        offset_emb: bool = False,
        scale_emb_gpu: bool = True,
        offset_emb_gpu: bool = True,
        center_canonical: bool = True,
        predict_mirror: bool = False,
        use_cache: bool = False
    ) -> None:
        super(SymEmbedding, self).__init__(
            in_dim, 
            emb_dim, 
            cube_scale, 
            n_obj, 
            n_keypoints, 
            predict_lrf, 
            prior, 
            scale_emb, 
            offset_emb, 
            scale_emb_gpu, 
            offset_emb_gpu,
            center_canonical,
            predict_mirror,
            use_cache
        )
        assert lrf_head is None or (pos_head is not None and self.predict_lrf)
        self.gpu = gpu
        self.init_emb()
        n_keypoints = self.n_keypoints
        if predict_mirror:
            n_keypoints = int(n_keypoints/2)
        self.decoder = define_mlp(d_in=self.hid_dim, d_out=(n_keypoints * (8 if self.predict_lrf else 3)) if pos_head is None else None, **decoder)
        self.pos_head = define_mlp(d_in=decoder.dims[-1], d_out=n_keypoints * 3, **pos_head) if pos_head is not None else None
        self.lrf_head = define_mlp(d_in=decoder.dims[-1], d_out=n_keypoints * 5, **lrf_head) if lrf_head is not None else None
    
    def init_emb(self, init_emb: Optional[Tensor] = None, std: Optional[float] = None):
        emb = FlexEmbedding(self.n_obj, self.hid_dim)
        self.emb = emb if self.gpu else [emb]   # hack to keep the Embedding on cpu despite of Pytorch-Lightning moving everything to GPUs
        if init_emb is not None:
            emb.weight = Parameter(init_emb)
        elif std is not None:
            init.normal_(emb.weight, std=std)

    def interpolate(self, idx: Tensor, num_steps: int) -> None:
        """
        Arguments
            idx: [n, 2]
        """
        n = len(idx)
        latents = self.access_cpu_or_gpu_emb(self.emb, idx.flatten(), self.gpu).view(n, 2, -1)  # [n, 2, hid_dim]
        step_weights = torch.linspace(0, 1, num_steps)[None, :, None]                           # [1, num_steps, 1]
        interpolated = (1 - step_weights) *  latents[:, :1] + step_weights * latents[:, 1:]     # [n, num_steps, hid_dim]
        self.n_obj = n * num_steps
        self.init_emb(interpolated.flatten(0, 1))
    
    def randomize(self, idx: Tensor, num_random: int, std: float) -> None:
        """
        Arguments
            idx: [n]
        """
        n = len(idx)
        self.n_obj = n * num_random
        self.init_emb(std=std)

    def copy_selection(self, idx: Tensor, num_copies: int) -> None:
        """
        Arguments
            idx: [n]
        """
        n = len(idx)
        latents = self.access_cpu_or_gpu_emb(self.emb, idx, self.gpu)     # [n, hid_dim]
        self.n_obj = n * num_copies
        self.init_emb(latents[:, None].expand(-1, num_copies, -1).flatten(0, 1))

    def encode(self, idx: Tensor) -> Tensor:
        """
        Arguments:
            idx: [B]
        Returns:
            latent: [B, hid_dim]
        """
        return self.access_cpu_or_gpu_emb(self.emb, idx, self.gpu)

    def decode(self, latent: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Arguments:
            latent: [L, hid_dim]
        Returns:
            pos: [L, n_kp, 3]
            optional lrf: [L, n_kp, 5]
        """
        L = latent.shape[0]
        decoded = self.decoder(latent)
        if self.pos_head is not None:
            pos = self.pos_head(decoded)
            if self.predict_mirror:
                pos = pos.view(L, int(self.n_keypoints/2), 3)
                ones = torch.ones_like(pos)
                ones[:,:,0] = -1.
                pos = torch.cat([pos, pos*ones], dim=1)
            pos = pos.view(L, self.n_keypoints, 3)
        else:
            if self.predict_mirror:
                pos = decoded.view(L, int(self.n_keypoints/2), -1)[..., :3]
                ones = torch.ones_like(pos)
                ones[:,:,0] = -1.
                decoded = torch.cat([pos, pos*ones], dim=1)
            pos = decoded.view(L, self.n_keypoints, 3)

        if self.predict_lrf:
            if self.lrf_head is not None:
                lrf = self.lrf_head(decoded).view(L, self.n_keypoints, 5)
            else:
                lrf = decoded.view(L, self.n_keypoints, -1)[..., 3:]
        else:
            lrf = None
        
        return pos, lrf
    
    # Ensures saving and loading embedding despite of hack for keeping it on CPU
    def get_extra_state(self):
        res = super(SymEmbedding, self).get_extra_state()
        emb = self.emb if self.gpu else self.emb[0]
        res["emb"] = emb.get_extra_state()
        return res
    
    def set_extra_state(self, state):
        super(SymEmbedding, self).set_extra_state(state)
        if state is not None and "emb" in state:
            emb = self.emb if self.gpu else self.emb[0]
            emb.set_extra_state(state["emb"])
