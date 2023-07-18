from typing import Iterable, Union

from pytorch_lightning.callbacks import BaseFinetuning
from torch.nn import Module, Embedding
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.optimizer import Optimizer

from .model import Model


class Freezer(BaseFinetuning):
    def __init__(
        self,
        detector: bool = False,
        kp_ids: bool = False,
        extractor: bool = False,
        global_extractor: bool = False,
        field: bool = False,
        train_bn: bool = False,
        train_emb: bool = True
    ):
        super().__init__()
        self.detector = detector
        self.kp_ids = kp_ids
        self.extractor = extractor
        self.global_extractor = global_extractor
        self.field = field
        self.train_bn = train_bn
        self.train_emb = train_emb

    @staticmethod
    def freeze(modules: Union[Module, Iterable[Union[Module, Iterable]]], train_bn: bool = True, train_emb: bool = True) -> None:
        """Freezes the parameters of the provided modules.

        Args:
            modules: A given module or an iterable of modules
            train_bn: If True, leave the BatchNorm layers in training mode
            train_emb: If True, do not freeze Embedding modules
        Returns:
            None
        """
        modules = BaseFinetuning.flatten_modules(modules)
        for mod in modules:
            if isinstance(mod, _BatchNorm) and train_bn:
                BaseFinetuning.make_trainable(mod)
            elif isinstance(mod, Embedding) and train_emb:
                BaseFinetuning.make_trainable(mod)
            else:
                BaseFinetuning.freeze_module(mod)
    
    @staticmethod
    def eval(modules: Union[Module, Iterable[Union[Module, Iterable]]], train_bn: bool = True, train_emb: bool = True) -> None:
        """Sets the provided modules to eval mode.

        Args:
            modules: A given module or an iterable of modules
            train_bn: If True, leave the BatchNorm layers in training mode
            train_emb: If True, leave Embedding modules in training mode
        Returns:
            None
        """
        modules = BaseFinetuning.flatten_modules(modules)
        for mod in modules:
            if isinstance(mod, _BatchNorm) and train_bn:
                mod.training = True
            elif isinstance(mod, Embedding) and train_emb:
                mod.training = True
            else:
                mod.training = False

    def freeze_before_training(self, model: Model) -> None:
        if getattr(model, "detector", None) is not None and self.detector:
            self.freeze(model.detector, self.train_bn, self.train_emb)
            self.eval(model.detector, self.train_bn, self.train_emb)
        if getattr(model, "kp_ids", None) is not None and self.kp_ids:
            model.kp_ids.requires_grad = False
        if getattr(model, "extractor", None) is not None and self.extractor:
            self.freeze(model.extractor, self.train_bn, self.train_emb)
            self.eval(model.extractor, self.train_bn, self.train_emb)
        if getattr(model, "global_extractor", None) is not None and self.global_extractor:
            self.freeze(model.global_extractor, self.train_bn, self.train_emb)
            self.eval(model.global_extractor, self.train_bn, self.train_emb)
        if getattr(model, "field", None) is not None and self.field:
            self.freeze(model.field, self.train_bn, self.train_emb)
            self.eval(model.field, self.train_bn, self.train_emb)

    def finetune_function(self, model: Model, epoch: int, optimizer: Optimizer, opt_idx: int) -> None:
        # Keep freezed modules in evaluation mode
        if getattr(model, "detector", None) is not None and self.detector:
            self.eval(model.detector, self.train_bn, self.train_emb)
        if getattr(model, "extractor", None) is not None and self.extractor:
            self.eval(model.extractor, self.train_bn, self.train_emb)
        if getattr(model, "global_extractor", None) is not None and self.global_extractor:
            self.eval(model.global_extractor, self.train_bn, self.train_emb)
        if getattr(model, "field", None) is not None and self.field:
            self.eval(model.field, self.train_bn, self.train_emb)