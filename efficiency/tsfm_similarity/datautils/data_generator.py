import torch
import numpy as np
from momentfm.data.synthetic_data import SyntheticDataset
from momentfm.data.anomaly_detection_dataset import AnomalyDetectionDataset
from momentfm.data.classification_dataset import ClassificationDataset
from momentfm.data.informer_dataset import InformerDataset


class DataGenerator:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def generate_synthetic_data(
        self,
        n_samples=1024,
        seq_len=512,
        n_channels=1,
        freq=1,
        freq_range=(1, 32),
        noise_mean=0.0,
        noise_std=0.1,
    ):
        synthetic_dataset = SyntheticDataset(
            n_samples=n_samples,
            seq_len=seq_len,
            freq=freq,
            freq_range=freq_range,
            noise_mean=noise_mean,
            noise_std=noise_std,
            random_seed=self.random_seed,
        )
        y, c = synthetic_dataset.gen_sinusoids_with_varying_freq()

        if n_channels > 1:
            y = y.repeat(1, n_channels, 1)
            for i in range(1, n_channels):
                y[:, i, :] += (
                    torch.randn(n_samples, seq_len) * noise_std
                )  # Add some variation to each channel

        return y, c

    def generate_random_data(
        self, n_samples=1024, seq_len=512, n_channels=1, distribution="normal", **kwargs
    ):
        if distribution == "normal":
            mean = kwargs.get("mean", 0)
            std = kwargs.get("std", 1)
            data = torch.randn(n_samples, n_channels, seq_len) * std + mean
        elif distribution == "uniform":
            low = kwargs.get("low", 0)
            high = kwargs.get("high", 1)
            data = torch.rand(n_samples, n_channels, seq_len) * (high - low) + low
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        return data

    def get_anomaly_detection_dataset(self, data_split="train", data_stride_len=512):
        dataset = AnomalyDetectionDataset(
            data_split=data_split,
            data_stride_len=data_stride_len,
            random_seed=self.random_seed,
        )
        return torch.tensor(dataset.data).unsqueeze(1), torch.tensor(dataset.labels)

    def get_classification_dataset(self, data_split="train"):
        dataset = ClassificationDataset(data_split=data_split)
        x = torch.tensor(dataset.data, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(dataset.labels, dtype=torch.long).unsqueeze(1)
        x = x.permute(2, 1, 0)
        return x, y

    def get_forecasting_dataset(
        self, forecast_horizon=192, data_split="train", data_stride_len=1
    ):
        dataset = InformerDataset(
            forecast_horizon=forecast_horizon,
            data_split=data_split,
            data_stride_len=data_stride_len,
            task_name="forecasting",
            random_seed=self.random_seed,
        )
        return (
            torch.tensor(dataset.data).permute(0, 2, 1),
            None,
        )  # No labels for forecasting task
