from __future__ import annotations

from typing import Any, Dict, Optional, Type

from .formatters import emit_formatted_output

ERROR_UNKNOWN = "unknown_error"
ERROR_NOT_FOUND = "resource_not_found"
ERROR_VALUE = "invalid_value"
ERROR_PERMISSION = "permission_denied"
ERROR_RUNTIME = "runtime_error"
ERROR_DEPENDENCY = "missing_dependency"

_EXCEPTION_CODE_MAP: Dict[Type[BaseException], str] = {
    FileNotFoundError: ERROR_NOT_FOUND,
    ValueError: ERROR_VALUE,
    PermissionError: ERROR_PERMISSION,
    RuntimeError: ERROR_RUNTIME,
    ImportError: ERROR_DEPENDENCY,
    KeyError: ERROR_VALUE,
}


def error_code_for_exception(exc: BaseException) -> str:
    for exc_type, code in _EXCEPTION_CODE_MAP.items():
        if isinstance(exc, exc_type):
            return code
    return ERROR_UNKNOWN


def emit_error(
    args,
    *,
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a standardized error payload or a human message."""

    text_message = message
    if details:
        extra = ", ".join(f"{key}={value}" for key, value in details.items())
        if extra:
            text_message = f"{message} ({extra})"

    error_entry: Dict[str, Any] = {"code": code, "message": message}
    if details:
        error_entry["details"] = details

    emit_formatted_output(
        args,
        success=False,
        message=message,
        data=details,
        errors=[error_entry],
        text=text_message,
        pretty=True,
    )


def emit_exception(
    args,
    exc: BaseException,
    *,
    message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    inferred_code = error_code_for_exception(exc)
    payload_details = dict(details or {})
    payload_details.setdefault("exception", exc.__class__.__name__)
    payload_details.setdefault("error", str(exc))
    emit_error(
        args,
        code=inferred_code,
        message=message or str(exc),
        details=payload_details,
    )
