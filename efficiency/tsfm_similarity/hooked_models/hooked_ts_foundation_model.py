from abc import ABC, abstractmethod
from typing import Any, List, Dict

import torch
from .hook import Hook
import torch.nn as nn
import logging


class HookedTSFoundationModel(ABC):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.model: nn.Module = self.setup_model(config)
        self.hooks: Dict[str, Hook] = self.setup_hooks()

    def __call__(self, x: torch.Tensor, *args, **kwds) -> torch.Tensor:
        return self.forward(x, *args, **kwds)

    # SETUP
    @abstractmethod
    def setup_model(self, config: Dict[str, Any]) -> nn.Module:
        """
        Setup the model.

        Args:
            config: Configuration dictionary.

        Returns:
            A ready-to-use PyTorch model.
        """
        raise NotImplementedError

    @abstractmethod
    def setup_hooks(self, backward: bool = False) -> Dict[str, Hook]:
        """
        Setup hooks for the model.

        Args:
            backward: Whether to setup backward hooks.

        Returns:
            A dictionary of hooks, with keys as the module names and values as the hooks.
        """
        raise NotImplementedError

    # INFERENCE
    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwds) -> torch.Tensor:
        """
        Still thinking about how to handle this, all models will for now have embeddings as an output.
        """
        pass

    @abstractmethod
    def extract_time_series_embeddings(
        self, x: torch.Tensor, *args, **kwds
    ) -> torch.Tensor:
        """
        Extract time series embeddings from the model.

        Args:
            x: Input tensor.

        Returns:
            Time series embeddings.
        """
        raise NotImplementedError

    # HOOK OPERATIONS
    def add_hook(self, name: str, hook: Hook) -> None:
        """
        Add a hook to the model.

        Args:
            hook: Hook to add.
        """
        self.hooks[name] = hook
        logging.debug(f"Added hook {name} to model")

    def remove_hook(self, name: str) -> None:
        """
        Remove a hook from the model.

        Args:
            name: Name of the hook to remove.
        """
        hook = self.hooks[name]
        hook.close()
        self._remove_hook(hook)
        logging.debug(f"Removed hook {name} from model")

    def clear_hooks(self) -> None:
        """
        Clear all hooks from the model.
        """
        for hook in self.hooks.values():
            hook.close()
            self._remove_hook(hook)
        logging.debug("Cleared all hooks from model")
