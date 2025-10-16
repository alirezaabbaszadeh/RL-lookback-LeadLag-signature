import numpy as np
import pytest

iisignature = pytest.importorskip("iisignature")

from models.leadlag.signature_extractor import SignatureConfig, SignatureExtractor


def _sample_pair() -> np.ndarray:
    t = np.linspace(0, 1, num=16)
    return np.column_stack((np.sin(t), np.cos(t)))


def test_signature_extractor_basic_value():
    extractor = SignatureExtractor(SignatureConfig(order=2, scaling_method="mean-centering"), cache_size=2)
    value = extractor.compute(_sample_pair())

    assert isinstance(value, float)
    assert not np.isnan(value)


def test_signature_extractor_batch_matches_individual():
    extractor = SignatureExtractor(SignatureConfig(order=2, scaling_method="mean-centering"), cache_size=2)
    batch = np.stack([_sample_pair(), _sample_pair() * 1.01])

    individual = [extractor.compute(batch[0]), extractor.compute(batch[1])]
    batch_values = extractor.compute_batch(batch)

    np.testing.assert_allclose(batch_values, np.array(individual))


def test_signature_extractor_validates_shape():
    extractor = SignatureExtractor(SignatureConfig(order=2, scaling_method="mean-centering"), cache_size=1)

    with pytest.raises(ValueError):
        extractor.compute(np.ones((4,)))
