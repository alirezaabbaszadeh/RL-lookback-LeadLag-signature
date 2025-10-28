"""Utilities for summarizing roadmap STATUS_TRACKER state."""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

ITEM_PATTERN = re.compile(
    r'ITEM\s*\{MODULE:\s*"(?P<module>[^"]+)",\s*ISSUE:\s*"(?P<issue>[^"]+)",\s*NEXT_STEP:\s*"(?P<next_step>[^"]+)"\}'
)

from leadlag.cli.formatters import add_format_flags, emit_formatted_output, finalize_format_args
from leadlag.reporting.logging_utils import get_logger, setup_logging


@dataclass
class StatusItem:
    module: str
    issue: str
    next_step: str


def _extract_open_items_section(text: str) -> str:
    """Return the raw block inside OPEN_ITEMS [ ... ]."""
    marker = "OPEN_ITEMS ["
    start_idx = text.find(marker)
    if start_idx == -1:
        raise ValueError("Could not locate OPEN_ITEMS section in roadmap document.")

    # Move to the character after '['.
    start_idx = text.find("[", start_idx) + 1
    if start_idx == 0:
        raise ValueError("Malformed OPEN_ITEMS section.")

    depth = 1
    end_idx = start_idx
    while end_idx < len(text) and depth > 0:
        char = text[end_idx]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                break
        end_idx += 1

    if depth != 0:
        raise ValueError("Unclosed OPEN_ITEMS bracket in roadmap document.")

    return text[start_idx:end_idx].strip()


def parse_status_items(text: str) -> List[StatusItem]:
    """Parse STATUS_TRACKER open items from the roadmap text."""
    section = _extract_open_items_section(text)
    items: List[StatusItem] = []
    for match in ITEM_PATTERN.finditer(section):
        items.append(
            StatusItem(
                module=match.group("module"),
                issue=match.group("issue"),
                next_step=match.group("next_step"),
            )
        )
    return items


def format_text(items: Sequence[StatusItem]) -> str:
    """Produce a human-readable summary."""
    if not items:
        return "No open roadmap items found."

    module_width = max(len("Module"), *(len(item.module) for item in items))
    issue_width = max(len("Issue"), *(len(item.issue) for item in items))

    header = f"{'Module'.ljust(module_width)}  {'Issue'.ljust(issue_width)}  Next Step"
    separator = "-" * len(header)

    lines = [header, separator]
    for item in items:
        lines.append(
            f"{item.module.ljust(module_width)}  {item.issue.ljust(issue_width)}  {item.next_step}"
        )
    return "\n".join(lines)


def collect_status(path: Path) -> List[StatusItem]:
    """Load roadmap file and return parsed STATUS_TRACKER items."""
    text = path.read_text(encoding="utf-8")
    return parse_status_items(text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize roadmap STATUS_TRACKER open items.")
    parser.add_argument(
        "--roadmap",
        type=Path,
        default=Path("docs/future_roadmap.pseudo"),
        help="Path to roadmap pseudo-document.",
    )
    add_format_flags(parser, default="text")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        help="Optional log file path.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    finalize_format_args(args, remove_in="0.2.0")

    log_path = args.log_path or Path("status_summary.log")
    setup_logging(log_path, level=str(args.log_level).upper(), context={"module": "status_summary"})
    logger = get_logger("reporting.status_summary", context={"roadmap": str(args.roadmap)})

    items = collect_status(args.roadmap)
    logger.info("Collected status items", context={"count": len(items)})

    payload = [asdict(item) for item in items]
    message = f"Collected {len(items)} status item(s)."
    emit_formatted_output(
        args,
        data=payload,
        text=format_text(items),
        message=message,
        pretty=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
