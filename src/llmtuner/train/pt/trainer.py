from types import MethodType
from typing import TYPE_CHECKING, Optional

from transformers import Trainer

from ...extras.logging import get_logger
from ..utils import create_custom_optimzer, create_custom_scheduler


if TYPE_CHECKING:
    import torch

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class CustomTrainer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(self, finetuning_args: "FinetuningArguments", **kwargs) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)
        self.register_hooks()  # Register hooks during initialization
    
    def register_hooks(self):
        def print_grad(grad):
            print(grad)
            raise RuntimeError("Exiting due to gradient inspection")

        for name, param in self.model.named_parameters():
            # print(name, param)
            # exit(0)
            if param.requires_grad:
                # param.register_hook(lambda grad, name=name: print(f"Gradient for {name}: {grad}"))
                param.register_hook(lambda grad, name=name: print(f"Gradient for {name}"))
                
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)
