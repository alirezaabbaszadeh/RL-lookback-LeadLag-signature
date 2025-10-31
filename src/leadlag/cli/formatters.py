from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def add_format_flags(parser: argparse.ArgumentParser, *, default: str = "text") -> None:
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default=default,
        help=f"Output format for CLI responses (default: {default}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="[DEPRECATED] Use --format json instead.",
    )

def finalize_format_args(args: argparse.Namespace, *, remove_in: str = "0.2.0") -> None:
    if getattr(args, "json", False):
        try:
            from leadlag.utils.deprecations import warn_flag_deprecated

            warn_flag_deprecated("--json", replacement="--format json", remove_in=remove_in)
        except Exception:
            pass
        setattr(args, "format", "json")
    # normalize convenience boolean
    setattr(args, "json", getattr(args, "format", "text") == "json")

def wants_json(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "json", False))

def to_json(payload: object, *, pretty: bool | int = False) -> str:
    if pretty:
        indent = 2 if pretty is True else int(pretty)
        return json.dumps(payload, indent=indent, ensure_ascii=False)
    return json.dumps(payload, ensure_ascii=False)

def _coerce_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return serialize_cli_args(value)
    if isinstance(value, dict):
        return {str(k): _coerce_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_json(v) for v in value]
    return value

def serialize_cli_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: _coerce_json(value) for key, value in vars(args).items()}

def _resolve_command(command: str | None) -> str:
    if command:
        return command
    exe = sys.argv[0] if sys.argv else "python"
    tail: Iterable[str] = sys.argv[1:] if len(sys.argv) > 1 else []
    parts = [exe, *tail]
    return " ".join(parts).strip()

def _build_envelope(
    args: argparse.Namespace,
    *,
    success: bool,
    data: Any = None,
    message: str | None = None,
    errors: list[dict[str, Any]] | None = None,
    artifacts: Any = None,
    command: str | None = None,
) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "success": bool(success),
        "command": _resolve_command(command),
        "args": serialize_cli_args(args),
        "format": getattr(args, "format", None),
        "errors": errors or [],
    }
    if message is not None:
        envelope["message"] = message
    if data is not None:
        envelope["data"] = data
    if artifacts is not None:
        envelope["artifacts"] = artifacts
    return envelope

def emit_formatted_output(
    args: argparse.Namespace,
    *,
    text: str | None = None,
    success: bool = True,
    data: Any = None,
    message: str | None = None,
    errors: list[dict[str, Any]] | None = None,
    artifacts: Any = None,
    pretty: bool | int = False,
    command: str | None = None,
) -> None:
    """Emit output respecting the selected CLI format."""
    if wants_json(args):
        payload = _build_envelope(
            args,
            success=success,
            data=data,
            message=message,
            errors=errors,
            artifacts=artifacts,
            command=command,
        )
        print(to_json(payload, pretty=pretty))
        return
    if text is not None:
        print(text)
