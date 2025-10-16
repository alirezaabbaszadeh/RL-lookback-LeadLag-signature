import numpy as np
import pytest

try:
    import pandas as pd
except ValueError:
    pytest.skip("Pandas binary incompatible with current numpy", allow_module_level=True)

from models.leadlag.signature_extractor import SIGNATURE_AVAILABLE
from models.leadlag.signature_feature_pipeline import (
    CompressionConfig,
    SignatureFeatureConfig,
    SignatureFeaturePipeline,
    TORCH_AVAILABLE,
)

if not SIGNATURE_AVAILABLE:
    pytest.skip("iisignature not installed", allow_module_level=True)


def _make_log_returns(seed: int) -> "pd.DataFrame":
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=32, freq="D")
    data = rng.normal(scale=0.01, size=(len(index), 4))
    return pd.DataFrame(data, index=index, columns=list("ABCD"))


def test_signature_pipeline_top_k_selection():
    cfg = SignatureFeatureConfig(
        orders=[2, 3],
        sig_method="custom",
        top_k=3,
        compression=CompressionConfig(method=None),
    )
    pipeline = SignatureFeaturePipeline(cfg)
    log_returns = _make_log_returns(0)
    pipeline.fit([log_returns])
    features = pipeline.transform_single(log_returns)

    assert len(features) == 3
    assert all(name.startswith("sig_o") for name in features.keys())


def test_signature_pipeline_pca_reduction():
    cfg = SignatureFeatureConfig(
        orders=[2],
        sig_method="custom",
        top_k=6,
        compression=CompressionConfig(method="pca", n_components=2),
    )
    pipeline = SignatureFeaturePipeline(cfg)
    windows = [_make_log_returns(seed) for seed in range(3)]
    transformed = pipeline.fit_transform(windows)

    assert transformed.shape == (3, 2)


def test_signature_pipeline_autoencoder_dependency():
    cfg = SignatureFeatureConfig(
        orders=[2],
        sig_method="custom",
        top_k=4,
        compression=CompressionConfig(method="autoencoder", n_components=2, autoencoder_epochs=1),
    )
    pipeline = SignatureFeaturePipeline(cfg)
    windows = [_make_log_returns(seed) for seed in range(2)]

    if TORCH_AVAILABLE:
        transformed = pipeline.fit_transform(windows)
        assert transformed.shape[1] == 2
    else:
        with pytest.raises(ImportError):
            pipeline.fit_transform(windows)
