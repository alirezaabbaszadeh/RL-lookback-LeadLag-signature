from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from leadlag.cli.formatters import (
    add_format_flags,
    emit_formatted_output,
    finalize_format_args,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a concise summary of a run_manifest.json file.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Run directory containing run_manifest.json or the manifest file itself.",
    )
    add_format_flags(parser, default="text")
    args = parser.parse_args(argv)
    finalize_format_args(args)
    return args


def _resolve_manifest_path(root: Path) -> Path:
    if root.is_dir():
        candidate = root / "run_manifest.json"
    else:
        candidate = root
    if not candidate.is_file():
        raise FileNotFoundError(f"No run_manifest.json found under {root}")
    return candidate


def _load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_none(value: Any, default: str = "NA") -> str:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return ",".join(str(item) for item in value)
    return str(value)


def _build_summary(manifest: Dict[str, Any], manifest_path: Path) -> tuple[str, Dict[str, Any]]:
    feature_time = manifest.get("feature_time", {})
    config_sources = manifest.get("config_sources", [])
    summary_data: Dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "run_id": manifest.get("run_id"),
        "seed": manifest.get("seed"),
        "window_index": manifest.get("window_index"),
        "env_steps_reported": manifest.get("env_steps_reported", manifest.get("requested_env_steps")),
        "env_steps_actual": manifest.get("env_steps_actual", manifest.get("actual_env_steps")),
        "feature_rows": feature_time.get("rows"),
        "feature_checked_rows": feature_time.get("checked_rows"),
        "feature_min_lag_ns": feature_time.get("min_lag_ns"),
        "feature_max_lag_ns": feature_time.get("max_lag_ns"),
        "feature_tz": feature_time.get("tz"),
        "feature_freq_hint": feature_time.get("freq_hint"),
        "config_sources": config_sources,
    }
    parts = [
        f"run_id={_coerce_none(summary_data['run_id'])}",
        f"seed={_coerce_none(summary_data['seed'])}",
        f"window={_coerce_none(summary_data['window_index'])}",
        f"env_steps_reported={_coerce_none(summary_data['env_steps_reported'])}",
        f"env_steps_actual={_coerce_none(summary_data['env_steps_actual'])}",
        f"feature_rows={_coerce_none(summary_data['feature_rows'])}",
        f"checked_rows={_coerce_none(summary_data['feature_checked_rows'])}",
        f"min_lag_ns={_coerce_none(summary_data['feature_min_lag_ns'])}",
        f"max_lag_ns={_coerce_none(summary_data['feature_max_lag_ns'])}",
        f"tz={_coerce_none(summary_data['feature_tz'])}",
        f"freq_hint={_coerce_none(summary_data['feature_freq_hint'])}",
        f"config_sources={_coerce_none(summary_data['config_sources'], default='NA')}",
    ]
    return " ".join(parts), summary_data


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest_path = _resolve_manifest_path(args.root)
        manifest = _load_manifest(manifest_path)
    except FileNotFoundError as exc:
        message = str(exc)
        emit_formatted_output(
            args,
            text=message,
            success=False,
            message=message,
            errors=[{"code": "manifest_not_found", "message": message}],
        )
        return 1
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        message = f"Failed to parse manifest at {args.root}: {exc}"
        emit_formatted_output(
            args,
            text=message,
            success=False,
            message=message,
            errors=[{"code": "invalid_manifest", "message": message}],
        )
        return 1

    summary_text, summary_data = _build_summary(manifest, manifest_path)
    emit_formatted_output(
        args,
        text=summary_text,
        success=True,
        data=summary_data,
        message="Run manifest summary.",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
