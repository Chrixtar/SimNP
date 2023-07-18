from typing import Dict, List, Optional, Tuple
import warnings

import torch
from torch import Tensor
from torch.nn import MultiheadAttention, Parameter, init
import torch.nn.functional as F

from .extractor import Extractor
from utils.model import define_mlp
from utils.flex_embedding import FlexEmbedding
from utils.positional_encoder import PositionalEncoder1D


class SymEmbedding(Extractor):
    def __init__(
        self,
        n_kp: int,
        kp_dim: int, 
        out_dim: int,
        n_obj: int,
        n_emb: int,
        a_dim: int,
        a_heads: int,
        use_ids_as_queries: bool = False,
        pos_encoder: Optional[dict] = None,
        optimize_scores_directly: bool = False,
        query_mlp: Optional[List[int]] = None,
        out_mlp: Optional[List[int]] = None,
        activation: str = "ReLU",
        layer_norm: bool = False,
        gpu: bool = True,
        init_std: float = 1.0
    ) -> None:
        super(SymEmbedding, self).__init__(n_kp, kp_dim, out_dim, n_obj)
        self.a_heads = a_heads
        self.a_emb_dim = a_heads * a_dim
        self.n_emb = n_emb
        self.gpu = gpu
        self.init_std = init_std
        self.init_emb()       # hack to keep the Embedding on cpu despite of Pytorch-Lightning moving everything to GPUs
        self.optimize_scores_directly = optimize_scores_directly
        if self.optimize_scores_directly:
            assert a_heads == 1, "Direct score optimization only implemented for a single head"
            assert pos_encoder is None, "Use of kp positions is incompatible with direct score optimization"
            assert not use_ids_as_queries, "Use of kp ids is incompatible with direct score optimization"
            self.scores = Parameter(torch.randn(self.n_kp, self.n_emb))
        else:
            self.keys = Parameter(torch.randn(n_emb, self.a_emb_dim))
            self.queries = Parameter(torch.randn(self.n_kp, self.a_emb_dim))
            query_dim = self.a_emb_dim
            self.use_ids_as_queries = use_ids_as_queries
            if self.use_ids_as_queries:
                query_dim += kp_dim
            if pos_encoder is not None:
                self.pos_enc = PositionalEncoder1D(**pos_encoder)
                query_dim += self.pos_enc.d_out(3)
            else:
                self.pos_enc = None
            if query_dim != self.a_emb_dim:
                self.query_mlp = define_mlp(query_mlp if query_mlp is not None else [], query_dim, self.a_emb_dim, activation, layer_norm)
            else:
                self.query_mlp = None
            self.attention = MultiheadAttention(self.a_emb_dim, a_heads, batch_first=True)
        if out_mlp is not None or self.a_emb_dim != self.out_dim:
            self.out_mlp = define_mlp(out_mlp if out_mlp is not None else [], self.a_emb_dim, self.out_dim, activation, layer_norm)
        else:
            self.out_mlp = None
    
    def init_emb(self, init_emb: Optional[Tensor] = None, std: Optional[float] = None):
        emb = FlexEmbedding(self.n_obj, self.n_emb * self.a_emb_dim)
        self.emb = emb if self.gpu else [emb]
        if init_emb is not None:
            emb.weight = Parameter(init_emb)
        else:
            init.normal_(emb.weight, std=self.init_std if std is None else std)

    def interpolate(self, idx: Tensor, num_steps: int, embedding_subset: Optional[Tensor] = None) -> None:
        """
        Arguments
            idx: [n, 2]
        """
        n = len(idx)
        latents = self.access_cpu_or_gpu_emb(self.emb, idx.flatten(), self.gpu).view(n, 2, self.n_emb, self.a_emb_dim)  # [n, 2, n_emb, a_emb_dim]
        step_weights = torch.linspace(0, 1, num_steps)[None, :, None, None]                                             # [1, num_steps, 1]
        interpolated = (1 - step_weights) *  latents[:, :1] + step_weights * latents[:, 1:]                             # [n, num_steps, n_emb, a_emb_dim]
        if embedding_subset is not None:
            assert embedding_subset.max() < self.n_emb, "Found embedding index in subset that is out of range"
            tmp = interpolated
            interpolated = latents[:, :1].expand(-1, num_steps, -1, -1).clone()
            interpolated[:, :, embedding_subset] = tmp[:, :, embedding_subset]
        self.n_obj = n * num_steps
        self.init_emb(interpolated.flatten(0, 1).flatten(-2, -1))
    
    def randomize(self, idx: Tensor, num_random: int, std: float, embedding_subset: Optional[Tensor] = None) -> None:
        """
        Arguments
            idx: [n]
        """
        n = len(idx)
        if embedding_subset is not None:
            assert embedding_subset.max() < self.n_emb, "Found embedding index in subset that is out of range"
            latents = self.access_cpu_or_gpu_emb(self.emb, idx, self.gpu).view(n, self.n_emb, self.a_emb_dim)   # [n, n_emb, a_emb_dim]
        self.n_obj = n * num_random
        self.init_emb(std=std)
        if embedding_subset is not None:
            random_latents = self.access_cpu_or_gpu_emb(self.emb, idx, self.gpu).view(n, self.n_emb, self.a_emb_dim)   # [n, n_emb, a_emb_dim]
            random_latents[:, embedding_subset] = latents[:, embedding_subset]
            self.init_emb(random_latents.flatten(0, 1))

    def copy_selection(self, idx: Tensor, num_copies: int) -> None:
        """
        Arguments
            idx: [n]
        """
        n = len(idx)
        latents = self.access_cpu_or_gpu_emb(self.emb, idx, self.gpu)     # [n, hid_dim]
        self.n_obj = n * num_copies
        self.init_emb(latents[:, None].expand(-1, num_copies, -1).flatten(0, 1))

    def forward(
        self,
        idx: Tensor,
        kp_pos: Tensor,
        kp_ids: Tensor,
        **kwargs
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Arguments:
            idx: [B]
            kp_pos: [B, n_kp, 3]
            kp_ids: [n_kp, kp_dim]
        Returns:
            feat: [B, n_kp, out_dim]
            inter_res: Dict with
                attn_map [B, n_kp, n_emb]
        """
        B = len(idx)
        v = self.access_cpu_or_gpu_emb(self.emb, idx, self.gpu).view(B, self.n_emb, -1)
        if self.optimize_scores_directly:
            map = F.softmax(self.scores, dim=-1)        # [n_kp, n_emb]
            feat = torch.matmul(map, v)                 # [B, n_kp, a_emb_dim]
            map = map.expand(B, self.n_kp, self.n_emb)  # [B, n_kp, n_emb]
        else:
            q = self.queries.expand(B, -1, -1)
            k = self.keys.expand(B, -1, -1)
            if self.use_ids_as_queries:
                q = torch.cat((q, kp_ids.expand(B, -1, -1)), dim=-1)
            if self.pos_enc is not None:
                q = torch.cat((q, self.pos_enc(kp_pos.detach())), dim=-1)
            if self.query_mlp is not None:
                q = self.query_mlp(q.flatten(0, 1)).view(-1, self.n_kp, self.a_emb_dim)
            # [B, n_kp, a_emb_dim], [B, n_kp, n_emb]
            feat, map = self.attention(q, k, v, need_weights=True)
        if self.out_mlp is not None:
            feat = self.out_mlp(feat.flatten(0, 1)).view(-1, self.n_kp, self.out_dim)
        return feat, {"attn_map": map}
    
    # Ensures saving and loading embedding despite of hack for keeping it on CPU
    def get_extra_state(self):
        emb = self.emb if self.gpu else self.emb[0]
        return {"emb": emb.get_extra_state()}
    
    def set_extra_state(self, state):
        if state is not None and "emb" in state:
            emb = self.emb if self.gpu else self.emb[0]
            emb.set_extra_state(state["emb"])
