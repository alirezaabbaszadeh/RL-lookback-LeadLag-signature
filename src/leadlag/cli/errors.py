from __future__ import annotations

from typing import Any, Dict, Optional

from .formatters import wants_json, to_json


def emit_error(
    args,
    *,
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a standardized error payload or a human message.

    JSON shape:
      {"error": {"code": str, "message": str, "details": {...}}}
    """
    if wants_json(args):
        payload: Dict[str, Any] = {"error": {"code": code, "message": message}}
        if details:
            payload["error"]["details"] = details
        print(to_json(payload))
    else:
        print(f"{message}")

