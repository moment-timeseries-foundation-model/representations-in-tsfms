from typing import Any, Dict, List
from .hooked_ts_foundation_model import HookedTSFoundationModel
from chronos import ChronosPipeline
from .hook import Hook
import torch.nn as nn
import torch
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", message="torch.utils._pytree._register_pytree_node is deprecated"
)
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    message=".*torch.utils._pytree._register_pytree_node.*"
)


class HookedChronos(HookedTSFoundationModel):
    """
    Hooked Chronos model, from Amazon Science.
    input shape: (B, C, L)
    WARNING - works only with univariate data, so C has to be 1.
    C is always collapsed in the process_input method.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config=config)

    def setup_model(self, config: Dict[str, Any]) -> nn.Module:
        chronos = ChronosPipeline.from_pretrained(config["model_version"])
        logging.info(f"Loaded Chronos variant {config['model_version']}")
        return chronos

    def setup_hooks(self, backward=False) -> Dict[str, Hook]:
        encoder_hooks = []
        for block in self.model.model.model.encoder.block:
            hook = Hook(block, backward=backward)
            encoder_hooks.append(hook)
        final_layer_norm = Hook(
            self.model.model.model.encoder.final_layer_norm, backward=backward
        )
        encoder_hooks.append(final_layer_norm)
        return {"encoder": encoder_hooks}

    def forward(self, x_BCL: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.extract_time_series_embeddings(x_BCL, *args, **kwargs)

    def process_input(self, x_BCL: torch.Tensor) -> torch.Tensor:
        if len(x_BCL.shape) != 3:
            raise ValueError("Bad input shape, expected (B, C, L)")
        if x_BCL.shape[1] != 1:
            raise ValueError(
                "Chronos model works only with univariate data, C has to be 1"
            )
        x_BL = x_BCL.squeeze(1)
        return x_BL

    def extract_time_series_embeddings(
        self, x_BCL: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Extract time series embeddings from the Chronos.
        ## inference function, processing the input
        Args:
            x: Input time series data with shape (B, C, L).

        Returns:
            Time series embeddings, with shape (B, L(513), D).
        Recommended to average along L axis to get a single embedding for each time series.
        """
        x_BL = self.process_input(x_BCL)
        embeddings, tokenizer_state = self.model.embed(x_BL)
        embeddings_BLD = embeddings
        return embeddings_BLD

    def get_encoder_representations(
        self, x_BCL: torch.Tensor, *args, **kwargs
    ) -> List[torch.Tensor]:
        """
        Get the encoder representation of the input. Specific for models with encoder.

        Args:
            x: Input time series data with shape (B, C, L).

        Returns:
            List of encoder representations, with shape (B, L(513), D).
        Recommended to average along L axis to get a single embedding for each time series.
        """
        self.forward(x_BCL, *args, **kwargs)
        representations = []
        for hook in self.hooks["encoder"]:
            level_representations_BLD = hook.output
            representations.append(level_representations_BLD)
        return representations
