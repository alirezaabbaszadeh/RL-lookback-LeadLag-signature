from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_KEYS = {"success", "command", "args", "format"}
OPTIONAL_KEYS = {"data", "message", "errors", "artifacts"}


def validate_payload(payload: dict) -> None:
    missing = REQUIRED_KEYS - payload.keys()
    if missing:
        raise SystemExit(f"Missing required keys: {sorted(missing)}")

    if not isinstance(payload["success"], bool):
        raise SystemExit("Field 'success' must be a boolean.")

    if not (payload.get("data") or payload.get("message")):
        raise SystemExit("Payload must include either 'data' or 'message'.")

    extra_keys = set(payload.keys()) - REQUIRED_KEYS - OPTIONAL_KEYS
    if extra_keys:
        raise SystemExit(f"Unexpected keys present: {sorted(extra_keys)}")

    if payload.get("errors") not in (None, []):
        raise SystemExit(f"Payload reports errors: {payload['errors']}")

    if payload["success"] is not True:
        raise SystemExit("Payload indicates failure (success != true).")


def load_payload(source: Path | None) -> dict:
    text: str
    if source is None:
        text = sys.stdin.read()
    else:
        text = source.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid JSON: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate leadlag CLI JSON payload schema.")
    parser.add_argument("path", nargs="?", type=Path, help="Path to JSON file (defaults to stdin).")
    args = parser.parse_args(argv)

    payload = load_payload(args.path)
    validate_payload(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
