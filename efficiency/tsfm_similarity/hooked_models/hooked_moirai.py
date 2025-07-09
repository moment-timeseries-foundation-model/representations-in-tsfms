from typing import Any, Dict, List
from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
from .hooked_ts_foundation_model import HookedTSFoundationModel
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


class HookedMoirai(HookedTSFoundationModel):
    """
    Hooked Moirai model, from Salesforce.
    Input shape: (B, C, L)
    Works with multivariate data, C is the number of channels.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config=config)

    def setup_model(self, config: Dict[str, Any]) -> nn.Module:
        module = MoiraiModule.from_pretrained(
            config["model_version"],
        )
        logging.info(f"Loaded Moirai variant {config['model_version']}")
        return module

    def setup_hooks(self, backward=False) -> Dict[str, Hook]:
        encoder_hooks = []
        for block in self.model.encoder.layers:
            hook = Hook(block, backward=backward)
            encoder_hooks.append(hook)
        final_layer_norm = Hook(self.model.encoder.norm, backward=backward)
        encoder_hooks.append(final_layer_norm)
        return {"encoder": encoder_hooks}

    def forecast(
        self,
        x_BCL: torch.Tensor,
        prediction_length: int = 1,
        context_length: int = None,
        patch_size: int = 32,  # be careful with patch size
        num_samples: int = 100,
        target_dim: int = 1,
        feat_dynamic_real_dim: int = 0,
        past_feat_dynamic_real_dim: int = 0,
    ) -> torch.Tensor:
        """
        WARNING - INPUT PROCESSING \n
        Forcast into the specified future and given context using Moirai.\n
        Args:
            x: Input time series data, with shape (B, C, L).
            prediction_length: Number of time steps to predict into the future.
            context_length: Number of time steps to use as context.
            patch_size: Size of the patch for the attention mechanism (auto for automatic selection, recommended).
            num_samples: Number of samples to draw from the posterior.
            target_dim: Dimension of the target variable.
            feat_dynamic_real_dim: Dimension of the dynamic real features.
            past_feat_dynamic_real_dim: Dimension of the past dynamic real features.

        Returns:
            Forecasted time series, with shape (B, num_samples, prediction_length, target_dim).
        """
        x_BLC = self.process_input(x_BCL)
        if context_length is None:
            context_length = x_BLC.shape[
                1
            ]  # assume that the whole provided time series is the context
        forecaster = MoiraiForecast(
            module=self.model,
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size=patch_size,
            num_samples=num_samples,
            target_dim=target_dim,
            feat_dynamic_real_dim=feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        )
        past_observed_target_BLC = torch.ones_like(
            x_BLC, dtype=torch.bool
        )  # assume that all the data is observed
        past_is_pad_BL = torch.zeros(
            x_BLC.shape[:2], dtype=torch.bool
        )  # TODO will need adjustments if we have time series with different lengths, now assuming no padding, all time series have the same length
        forecast_B_num_samples_horizon_C = forecaster(
            past_target=x_BLC,
            past_observed_target=past_observed_target_BLC,
            past_is_pad=past_is_pad_BL,
        )
        if (
            len(forecast_B_num_samples_horizon_C.shape) == 3
        ):  # if only one channel is provided
            forecast_B_num_samples_horizon_C = (
                forecast_B_num_samples_horizon_C.unsqueeze(-1)
            )
        return forecast_B_num_samples_horizon_C

    def forward(self, x_BCL: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.forecast(x_BCL, *args, **kwargs)

    def process_input(self, x_BCL: torch.Tensor) -> torch.Tensor:
        if len(x_BCL.shape) != 3:
            raise ValueError("Bad input shape, expected (B, C, L)")
        x_BLC = x_BCL.permute(0, 2, 1)
        return x_BLC

    def extract_time_series_embeddings(
        self, x_BCL: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Extract time series embeddings from Moirai.
        Args:
            x: Input time series data, with shape (B, C, L).

        Returns:
            Time series embeddings, with shape (B, P*C, D),
        """
        encoder_representations = self.get_encoder_representation(x_BCL)
        last_hidden_state_BPaCD = encoder_representations[-1]
        return last_hidden_state_BPaCD

    def get_encoder_representations(
        self, x_BCL: torch.Tensor, *args, **kwargs
    ) -> List[torch.Tensor]:
        """
        Get the encoder representation of the input.

        Args:
            x: Input time series data, with shape (B, C, L).

        Returns:
            List of encoder representations, each with with shape (B, P*C, D),
        Recommended to average along P axis to get a single embedding for each time series or set of time series (if multiple channels are provided)
        """
        predictions_B_num_samples_horizon_C = self.forecast(x_BCL, *args, **kwargs)
        representations = []
        for hook in self.hooks["encoder"]:
            level_representations_BPaCD = hook.output
            representations.append(level_representations_BPaCD)
        return representations
