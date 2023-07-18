from typing import Optional

from torch import Tensor
from torch.nn import Embedding, Module, Parameter, init

from utils.flex_embedding import FlexEmbedding


class PinnedEmbedding(Module):
    def __init__(
        self,
        num_embeddings: int, 
        embedding_dim: int,
        gpu: bool = True,
        flex: bool = True,
        **kwargs
    ) -> None:
        super(PinnedEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.flex = flex
        self.emb_kwargs = kwargs
        self.gpu = gpu
        self.init()

    def init(
        self, 
        weight: Optional[Tensor] = None, 
        std: Optional[float] = None, 
        constant: Optional[float] = None
    ) -> None:
        if self.flex:
            emb = FlexEmbedding(self.num_embeddings, self.embedding_dim, **self.emb_kwargs)
        else:
            emb = Embedding(self.num_embeddings, self.embedding_dim, **self.emb_kwargs)
        self.emb = emb if self.gpu else [emb]   # hack to keep the Embedding on cpu despite of Pytorch-Lightning moving everything to GPUs
        if weight is not None:
            assert std is None and constant is None
            emb.weight = Parameter(weight)
        elif std is not None:
            assert constant is None
            init.normal_(emb.weight, std=std)
        elif constant is not None:
            init.constant_(emb.weight, constant)

    def forward(self, idx: Tensor) -> Tensor:
        device = idx.device
        if self.gpu:
            emb = self.emb
        else:
            emb = self.emb[0]
            idx = idx.cpu()
        return emb(idx).to(device=device)

    def get_extra_state(self):
        emb = self.emb if self.gpu else self.emb[0]
        return {"emb": emb.get_extra_state()}
    
    def set_extra_state(self, state):
        if state is not None and "emb" in state:
            emb = self.emb if self.gpu else self.emb[0]
            emb.set_extra_state(state["emb"])
