import numpy as np
import pytest

from models.leadlag.signature_extractor import (
    SIGNATURE_AVAILABLE,
    SignatureConfig,
    SignatureExtractor,
)


def _sample_pair(n: int = 16) -> np.ndarray:
    t = np.linspace(0, 1, num=n)
    return np.column_stack((np.sin(t), np.cos(t)))


def test_signature_extractor_basic_value():
    extractor = SignatureExtractor(
        SignatureConfig(order=2, scaling_method="mean-centering", allow_fallback=True),
        cache_size=2,
    )
    value = extractor.compute(_sample_pair())

    assert isinstance(value, float)
    assert not np.isnan(value)


def test_signature_extractor_batch_matches_individual():
    extractor = SignatureExtractor(
        SignatureConfig(order=2, scaling_method="mean-centering", allow_fallback=True),
        cache_size=2,
    )
    batch = np.stack([_sample_pair(), _sample_pair() * 1.01])

    individual = [extractor.compute(batch[0]), extractor.compute(batch[1])]
    batch_values = extractor.compute_batch(batch)

    np.testing.assert_allclose(batch_values, np.array(individual))


def test_signature_extractor_validates_shape():
    extractor = SignatureExtractor(
        SignatureConfig(order=2, scaling_method="mean-centering", allow_fallback=True),
        cache_size=1,
    )

    with pytest.raises(ValueError):
        extractor.compute(np.ones((4,)))


def test_signature_scale_invariance_on_noise():
    """Verify that expected |Levy area| does not artificially explode as L grows from 15 to 120."""
    extractor = SignatureExtractor(
        SignatureConfig(order=2, sig_method="levy", allow_fallback=True),
        cache_size=0,
    )
    rng = np.random.default_rng(42)

    # Average score over multiple independent trials
    trials = 50
    score_short = []
    score_long = []

    for _ in range(trials):
        noise_short = rng.standard_normal((15, 2))
        noise_long = rng.standard_normal((120, 2))
        score_short.append(abs(extractor.compute(noise_short)))
        score_long.append(abs(extractor.compute(noise_long)))

    mean_short = float(np.mean(score_short))
    mean_long = float(np.mean(score_long))

    # Without 1/sqrt(L) scaling, mean_long would be 8x larger than mean_short.
    # With 1/sqrt(L) scaling, both are of the same order of magnitude (ratio between 0.3 and 3.0).
    ratio = mean_long / max(1e-8, mean_short)
    assert 0.2 < ratio < 4.0, f"Scale ratio {ratio} indicates lack of scale invariance"


def test_signature_detects_synthetic_leader():
    """Verify that if col 0 leads col 1, Score(col 0 -> col 1) is strictly positive."""
    extractor = SignatureExtractor(
        SignatureConfig(order=2, sig_method="levy", allow_fallback=True),
        cache_size=0,
    )
    rng = np.random.default_rng(123)

    L = 60
    x = rng.standard_normal(L)
    # y lags x by 1 step with small independent noise
    y = np.zeros(L)
    y[1:] = x[:-1] + 0.1 * rng.standard_normal(L - 1)

    data_pair = np.column_stack((x, y))
    score = extractor.compute(data_pair)

    assert score > 0.0, f"Expected positive score for true leader, got {score}"
