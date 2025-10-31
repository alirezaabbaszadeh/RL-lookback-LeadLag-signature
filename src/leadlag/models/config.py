"""Configuration helpers for the lead-lag analyzer."""

from __future__ import annotations

import importlib.util
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, Literal, Optional, Union

SIGNATURE_AVAILABLE = importlib.util.find_spec("iisignature") is not None
P_TQDM_AVAILABLE = importlib.util.find_spec("p_tqdm") is not None

try:
    import dcor  # type: ignore

    DCOR_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    DCOR_AVAILABLE = False

try:
    from sklearn.feature_selection import mutual_info_classif  # type: ignore

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SKLEARN_AVAILABLE = False


@dataclass
class LeaderFollowerConfig:
    """Configuration class for Leader-Follower detection methods."""

    method: Literal["percentile"] = "percentile"

    # Percentile method parameters
    top_percentile: float = 50.0
    bottom_percentile: float = 50.0
    agg_func: str = "sum"

    def __post_init__(self) -> None:
        """Validate and set default cluster assignments."""

        if self.method == "percentile":
            if not (0 <= self.top_percentile <= 100):
                raise ValueError("top_percentile must be between 0 and 100")
            if not (0 <= self.bottom_percentile <= 100):
                raise ValueError("bottom_percentile must be between 0 and 100")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LeaderFollowerConfig":
        """Create LeaderFollowerConfig from dictionary input."""

        method = config_dict.get("method", "percentile")
        if method == "percentile":
            return cls(
                method=method,
                agg_func=config_dict.get("agg_func", "sum"),
                top_percentile=config_dict.get("top_percentile", 50.0),
                bottom_percentile=config_dict.get("bottom_percentile", 50.0),
            )

        raise ValueError(f"Unknown method: {method}")


@dataclass
class LeadLagConfig:
    """Configuration parameters for lead-lag analysis."""

    method: Literal["ccf_at_lag", "ccf_auc", "signature", "ccf_at_max_lag"] = "signature"
    correlation_method: Literal[
        "pearson", "kendall", "spearman", "distance", "mutual_information", "squared_pearson"
    ] = "pearson"
    lookback: Optional[int] = 252
    update_freq: Optional[int] = 1
    use_parallel: bool = True
    num_cpus: int = 7
    quantiles: int = 4
    show_progress: bool = True
    Scaling_Method: str = "mean-centering"
    sig_method: str = "levy"

    # Method-specific parameters
    lag: Optional[int] = None
    max_lag: Optional[int] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LeadLagConfig":
        """Create LeadLagConfig from dictionary input."""

        method = config_dict.get("method", "ccf_at_lag")
        if method == "dtw":
            raise ValueError(
                "DTW mode is deprecated for this project. Use 'signature' or other supported methods."
            )

        method_config = config_dict.get(method, {})

        common_params = {
            "method": method,
            "lookback": config_dict.get("lookback", 252),
            "update_freq": config_dict.get("update_freq", 1),
            "use_parallel": config_dict.get("use_parallel", True),
            "num_cpus": config_dict.get("num_cpus", 7),
            "show_progress": config_dict.get("show_progress", True),
            "Scaling_Method": config_dict.get("Scaling_Method", "mean-centering"),
        }

        if method == "ccf_at_lag":
            common_params["lag"] = method_config.get("lag", 1)
            common_params["correlation_method"] = method_config.get("correlation_method", "pearson")
            common_params["quantiles"] = method_config.get("quantiles", 4)
        elif method == "ccf_auc":
            common_params["max_lag"] = method_config.get("max_lag", 10)
            common_params["correlation_method"] = method_config.get("correlation_method", "pearson")
            common_params["quantiles"] = method_config.get("quantiles", 4)
        elif method == "ccf_at_max_lag":
            common_params["max_lag"] = method_config.get("max_lag", 10)
            common_params["correlation_method"] = method_config.get("correlation_method", "pearson")
            common_params["quantiles"] = method_config.get("quantiles", 4)
        elif method == "signature":
            common_params["correlation_method"] = method_config.get("correlation_method", "pearson")
            common_params["quantiles"] = method_config.get("quantiles", 4)
            common_params["sig_method"] = method_config.get("sig_method", "custom")

        return cls(**common_params)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""

        if self.method in ["ccf_at_lag"] and self.lag is None:
            raise ValueError("lag parameter is required for ccf_at_lag method")

        if self.method in ["ccf_auc", "ccf_at_max_lag"] and self.max_lag is None:
            raise ValueError("max_lag parameter is required for ccf_auc and ccf_at_max_lag methods")

        if self.method == "signature" and not SIGNATURE_AVAILABLE:
            raise ValueError("iisignature package is required for signature method")

        if self.correlation_method == "distance" and not DCOR_AVAILABLE:
            raise ValueError("dcor package is required for distance correlation method")

        if self.correlation_method == "mutual_information" and not SKLEARN_AVAILABLE:
            raise ValueError("scikit-learn is required for mutual information correlation")

        if self.use_parallel and not P_TQDM_AVAILABLE:
            self.use_parallel = False
            warnings.warn("Parallel processing disabled due to missing p_tqdm package")


def coerce_lead_lag_config(config: Union[LeadLagConfig, Dict[str, Any], Any]) -> LeadLagConfig:
    """Normalize various configuration inputs into a LeadLagConfig."""

    if isinstance(config, LeadLagConfig):
        return config
    if isinstance(config, dict):
        return LeadLagConfig.from_dict(config)
    if is_dataclass(config):
        return LeadLagConfig(**asdict(config))
    raise TypeError("config must be a LeadLagConfig instance or compatible dataclass/dict")


def coerce_leader_follower_config(
    config: Union[LeaderFollowerConfig, Dict[str, Any]]
) -> LeaderFollowerConfig:
    """Normalize configuration for the leader/follower detector."""

    if isinstance(config, LeaderFollowerConfig):
        return config
    if isinstance(config, dict):
        return LeaderFollowerConfig.from_dict(config)
    raise TypeError("method_config must be a dictionary or LeaderFollowerConfig instance")


__all__ = [
    "LeadLagConfig",
    "LeaderFollowerConfig",
    "coerce_lead_lag_config",
    "coerce_leader_follower_config",
    "SIGNATURE_AVAILABLE",
    "DCOR_AVAILABLE",
    "SKLEARN_AVAILABLE",
    "P_TQDM_AVAILABLE",
]
