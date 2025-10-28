from __future__ import annotations

import argparse
import json


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

