import logging
from typing import Any, Dict
import torch.nn as nn
import torch


class Hook:
    """
    Hook class to store the input and output of a torch module.

    Args:
        module: The module to hook.
        backward: Whether to hook the backward pass.
    """

    def __init__(self, module: nn.Module, backward: bool = False) -> None:
        self.input = None
        self.output = None
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
            logging.debug(f"Forward hook registered for module: {module}")
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
            logging.debug(f"Backward hook registered for module: {module}")

    def hook_fn(
        self, module: nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> None:
        """
        Hook function to store the input and output of the module.

        Args:
            module: The module to hook.
            input: The input tensor.
            output: The output tensor.
        """
        logging.debug(f"Hooked module {module}")
        logging.debug(f"Input: {input}")
        logging.debug(f"Output: {output}")
        output = output[0] if isinstance(output, tuple) else output
        input = input[0] if isinstance(input, tuple) else input
        self.input = input
        self.output = output

    def close(self) -> None:
        """
        Close the hook.
        """
        self.hook.remove()
        logging.debug(f"Hook removed for module: {self.hook.module}")
