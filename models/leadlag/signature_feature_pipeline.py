from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .matrix_builder import build_matrix
from .signature_extractor import SignatureConfig, SignatureExtractor

try:  # Optional dependency; autoencoder compression requires torch
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


@dataclass
class CompressionConfig:
    method: Optional[str] = None  # 'pca' or 'autoencoder'
    n_components: Optional[int] = None
    autoencoder_hidden_dims: List[int] = field(default_factory=lambda: [32])
    autoencoder_epochs: int = 50
    autoencoder_batch_size: int = 32
    autoencoder_learning_rate: float = 1e-3

    @classmethod
    def from_dict(cls, data: Dict[str, Union[str, int, float, List[int]]]) -> "CompressionConfig":
        if data is None:
            return cls()
        method = data.get("method")
        n_components = data.get("n_components")
        hidden = data.get("autoencoder_hidden_dims", [32])
        epochs = int(data.get("autoencoder_epochs", 50))
        batch_size = int(data.get("autoencoder_batch_size", 32))
        lr = float(data.get("autoencoder_learning_rate", 1e-3))
        return cls(
            method=method,
            n_components=int(n_components) if n_components is not None else None,
            autoencoder_hidden_dims=list(hidden),
            autoencoder_epochs=epochs,
            autoencoder_batch_size=batch_size,
            autoencoder_learning_rate=lr,
        )


@dataclass
class SignatureFeatureConfig:
    orders: List[int] = field(default_factory=lambda: [2])
    sig_method: str = "custom"
    scaling_method: str = "mean-centering"
    top_k: Optional[int] = 10
    min_magnitude: float = 0.0
    compression: CompressionConfig = field(default_factory=CompressionConfig)

    @classmethod
    def from_dict(cls, data: Dict) -> "SignatureFeatureConfig":
        orders = data.get("orders") or [2]
        if isinstance(orders, int):
            orders = [orders]
        sig_method = data.get("sig_method", "custom")
        scaling_method = data.get("scaling_method", "mean-centering")
        top_k = data.get("top_k", 10)
        min_magnitude = float(data.get("min_magnitude", 0.0))
        compression_cfg = CompressionConfig.from_dict(data.get("compression"))
        return cls(
            orders=list(map(int, orders)),
            sig_method=sig_method,
            scaling_method=scaling_method,
            top_k=None if top_k is None else int(top_k),
            min_magnitude=min_magnitude,
            compression=compression_cfg,
        )


def _flatten_matrix(matrix: pd.DataFrame, order: int) -> Dict[str, float]:
    features: Dict[str, float] = {}
    columns = list(matrix.columns)
    for i, asset_i in enumerate(columns):
        for j in range(i + 1, len(columns)):
            asset_j = columns[j]
            value = matrix.iloc[i, j]
            if pd.isna(value):
                continue
            features[f"sig_o{order}_{asset_i}_{asset_j}"] = float(value)
    return features


if TORCH_AVAILABLE:  # pragma: no cover - torch path covered when available
    class _TorchAutoencoder(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Sequence[int]) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            prev_dim = input_dim
            for hidden in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden))
                layers.append(nn.ReLU())
                prev_dim = hidden
            self.encoder = nn.Sequential(*layers, nn.Linear(prev_dim, latent_dim))

            decoder_layers: List[nn.Module] = []
            prev_dim = latent_dim
            for hidden in reversed(hidden_dims):
                decoder_layers.append(nn.Linear(prev_dim, hidden))
                decoder_layers.append(nn.ReLU())
                prev_dim = hidden
            decoder_layers.append(nn.Linear(prev_dim, input_dim))
            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed, latent
else:  # pragma: no cover - placeholder when torch absent
    class _TorchAutoencoder:  # type: ignore[too-many-ancestors]
        pass


class SignatureFeaturePipeline:
    """Generate signature-based pair features with optional compression."""

    def __init__(self, config: SignatureFeatureConfig) -> None:
        self.config = config
        self._extractors: Dict[int, SignatureExtractor] = {
            order: SignatureExtractor(SignatureConfig(order=order, scaling_method=config.scaling_method, sig_method=config.sig_method))
            for order in config.orders
        }
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._ae_model: Optional[_TorchAutoencoder] = None
        self._ae_device: Optional["torch.device"] = None
        self._selected_features: Optional[List[str]] = None
        self._all_features: Optional[List[str]] = None
        self._fitted = False

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SignatureFeaturePipeline":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - triggered when pyyaml missing
            raise ImportError("PyYAML is required to load signature feature configs from YAML") from exc

        with open(path, "r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        config = SignatureFeatureConfig.from_dict(data)
        return cls(config)

    def fit(self, log_returns_batch: Sequence[pd.DataFrame]) -> "SignatureFeaturePipeline":
        feature_frame = self._build_feature_frame(log_returns_batch)
        self._all_features = feature_frame.columns.tolist()
        self._selected_features = self._select_top_k(feature_frame)
        selected_frame = feature_frame[self._selected_features]
        self._fit_compression(selected_frame)
        self._fitted = True
        return self

    def fit_transform(self, log_returns_batch: Sequence[pd.DataFrame]) -> np.ndarray:
        self.fit(log_returns_batch)
        return self.transform(log_returns_batch)

    def transform(self, log_returns_batch: Sequence[pd.DataFrame]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SignatureFeaturePipeline must be fitted before calling transform()")
        feature_frame = self._build_feature_frame(log_returns_batch)
        if self._all_features is None or self._selected_features is None:
            raise RuntimeError("Pipeline is not properly fitted")
        feature_frame = feature_frame.reindex(columns=self._all_features, fill_value=0.0)
        selected = feature_frame[self._selected_features]
        return self._apply_compression(selected)

    def transform_single(self, log_returns: pd.DataFrame) -> Dict[str, float]:
        """Generate selected features for a single window (without compression)."""
        feature_frame = self._build_feature_frame([log_returns])
        features = feature_frame.iloc[0]
        if self._selected_features is None:
            selected = features
        else:
            selected = features[self._selected_features].fillna(0.0)
        return selected.to_dict()

    def _build_feature_frame(self, log_returns_batch: Sequence[pd.DataFrame]) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for log_returns in log_returns_batch:
            features = self._extract_features_for_orders(log_returns)
            rows.append(features)
        frame = pd.DataFrame(rows).fillna(0.0)
        if not frame.columns.tolist():
            return frame
        if self.config.min_magnitude > 0:
            mask = frame.abs().max(axis=0) >= self.config.min_magnitude
            frame = frame.loc[:, mask]
        return frame

    def _extract_features_for_orders(self, log_returns: pd.DataFrame) -> Dict[str, float]:
        features: Dict[str, float] = {}
        for order, extractor in self._extractors.items():
            matrix = build_matrix(log_returns, extractor.compute)
            features.update(_flatten_matrix(matrix, order))
        return features

    def _select_top_k(self, frame: pd.DataFrame) -> List[str]:
        if self.config.top_k is None or self.config.top_k <= 0 or frame.empty:
            return frame.columns.tolist()
        scores = frame.abs().mean(axis=0)
        ranked = scores.sort_values(ascending=False)
        return ranked.head(self.config.top_k).index.tolist()

    # ------------------------------------------------------------------
    # Compression helpers
    # ------------------------------------------------------------------
    def _fit_compression(self, frame: pd.DataFrame) -> None:
        cfg = self.config.compression
        if not cfg or not cfg.method:
            return
        if cfg.method.lower() == "pca":
            n_components = cfg.n_components or min(frame.shape[1], max(1, frame.shape[1] // 2))
            self._scaler = StandardScaler()
            scaled = self._scaler.fit_transform(frame.values)
            self._pca = PCA(n_components=n_components, random_state=0)
            self._pca.fit(scaled)
        elif cfg.method.lower() == "autoencoder":
            if not TORCH_AVAILABLE:
                raise ImportError("torch is required for autoencoder compression but is not installed")
            n_components = cfg.n_components or min(frame.shape[1], max(1, frame.shape[1] // 2))
            self._scaler = StandardScaler()
            scaled = self._scaler.fit_transform(frame.values)
            self._init_autoencoder(frame.shape[1], n_components, cfg)
            self._train_autoencoder(scaled, cfg)
        else:
            raise ValueError(f"Unknown compression method '{cfg.method}'")

    def _apply_compression(self, frame: pd.DataFrame) -> np.ndarray:
        if self._pca is not None and self._scaler is not None:
            scaled = self._scaler.transform(frame.values)
            return self._pca.transform(scaled)
        if self._ae_model is not None and self._scaler is not None:
            scaled = self._scaler.transform(frame.values)
            with torch.no_grad():  # type: ignore[attr-defined]
                tensor = torch.tensor(scaled, dtype=torch.float32, device=self._ae_device)  # type: ignore[attr-defined]
                _, latent = self._ae_model(tensor)
                return latent.cpu().numpy()
        return frame.values

    def _init_autoencoder(self, input_dim: int, latent_dim: int, cfg: CompressionConfig) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[attr-defined]
        self._ae_device = device
        hidden_dims = cfg.autoencoder_hidden_dims or [max(4, latent_dim * 2)]
        self._ae_model = _TorchAutoencoder(input_dim, latent_dim, hidden_dims).to(device)

    def _train_autoencoder(self, scaled: np.ndarray, cfg: CompressionConfig) -> None:
        assert self._ae_model is not None
        dataset = TensorDataset(torch.tensor(scaled, dtype=torch.float32))  # type: ignore[attr-defined]
        loader = DataLoader(dataset, batch_size=min(cfg.autoencoder_batch_size, len(dataset)), shuffle=True)  # type: ignore[attr-defined]
        optimizer = torch.optim.Adam(self._ae_model.parameters(), lr=cfg.autoencoder_learning_rate)  # type: ignore[attr-defined]
        loss_fn = nn.MSELoss()  # type: ignore[attr-defined]
        self._ae_model.train()
        for _ in range(cfg.autoencoder_epochs):
            for (batch,) in loader:
                batch = batch.to(self._ae_device)
                optimizer.zero_grad()
                reconstructed, _ = self._ae_model(batch)
                loss = loss_fn(reconstructed, batch)
                loss.backward()
                optimizer.step()


def load_signature_feature_pipeline(path: Union[str, Path]) -> SignatureFeaturePipeline:
    """Utility to load pipeline configuration from YAML path."""
    return SignatureFeaturePipeline.from_yaml(path)
