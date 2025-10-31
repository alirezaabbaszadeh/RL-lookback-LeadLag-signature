"""Compatibility shim for legacy ``models.leadlag`` imports."""

from __future__ import annotations

try:  # Prefer the installed package layout when available.
    from leadlag.models.leadlag import *  # type: ignore  # noqa: F401,F403
except ModuleNotFoundError:
    # Fall back to the local source-tree modules so that the legacy namespace
    # remains usable without requiring ``src`` on ``sys.path`` first.
    from .matrix_builder import build_matrices_batch, build_matrix  # noqa: F401
    from .signature_extractor import SignatureConfig, SignatureExtractor  # noqa: F401
    from .signature_feature_pipeline import (  # noqa: F401
        SignatureFeatureConfig,
        SignatureFeaturePipeline,
        load_signature_feature_pipeline,
    )
    from .window_processor import WindowProcessor  # noqa: F401

    __all__ = [
        "build_matrices_batch",
        "build_matrix",
        "SignatureConfig",
        "SignatureExtractor",
        "SignatureFeatureConfig",
        "SignatureFeaturePipeline",
        "WindowProcessor",
        "load_signature_feature_pipeline",
    ]
