from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from leadlag.cli.formatters import add_format_flags, emit_formatted_output, finalize_format_args
from leadlag.reporting.data_access import (
    ScenarioAggregate,
    discover_aggregate_dirs,
    load_aggregates,
)
from leadlag.reporting.logging_utils import get_logger, setup_logging
from leadlag.reporting.report_builder import ReportArtifacts, ReportBuilder


def chunk_for_pdf(text: str, lines_per_page: int = 44) -> List[List[str]]:
    raw_lines = text.splitlines()
    pages: List[List[str]] = []
    for start in range(0, len(raw_lines), lines_per_page):
        chunk = raw_lines[start : start + lines_per_page]
        pages.append(chunk or [" "])
    if not pages:
        pages.append([" "])
    return pages


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf_stream(lines: Sequence[str]) -> bytes:
    commands = ["BT", "/F1 12 Tf", "14 TL", "50 780 Td"]
    first_line = True
    for raw in lines:
        line = escape_pdf_text(raw)
        if not first_line:
            commands.append("T*")
        if not line:
            commands.append("( ) Tj")
        else:
            commands.append(f"({line}) Tj")
        first_line = False
    commands.append("ET")
    stream = "\n".join(commands) + "\n"
    return stream.encode("utf-8")


def write_simple_pdf(pages: Sequence[Sequence[str]], output_path: Path) -> None:
    content_streams = [build_pdf_stream(page) for page in pages]
    num_pages = len(content_streams)
    font_obj_num = 3 + num_pages * 2
    pdf_parts: List[bytes] = [b"%PDF-1.4\n"]
    offsets: List[int] = []

    def append_object(obj: bytes) -> None:
        offset = sum(len(part) for part in pdf_parts)
        offsets.append(offset)
        pdf_parts.append(obj)

    catalog = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    append_object(catalog)

    kids_refs = " ".join(f"{3 + idx * 2} 0 R" for idx in range(num_pages))
    pages_obj = (
        f"2 0 obj\n<< /Type /Pages /Count {num_pages} /Kids [ {kids_refs} ] >>\nendobj\n".encode(
            "utf-8"
        )
    )
    append_object(pages_obj)

    for idx, stream in enumerate(content_streams):
        page_obj_num = 3 + idx * 2
        content_obj_num = page_obj_num + 1
        page_obj = (
            f"{page_obj_num} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> "
            f"/Contents {content_obj_num} 0 R >>\n"
            "endobj\n"
        ).encode("utf-8")
        append_object(page_obj)

        content_obj = (
            f"{content_obj_num} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("utf-8")
            + stream
            + b"endstream\nendobj\n"
        )
        append_object(content_obj)

    font_obj = (
        f"{font_obj_num} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    ).encode("utf-8")
    append_object(font_obj)

    xref_offset = sum(len(part) for part in pdf_parts)
    xref_header = f"xref\n0 {len(offsets) + 1}\n0000000000 65535 f \n".encode("utf-8")
    pdf_parts.append(xref_header)
    for offset in offsets:
        pdf_parts.append(f"{offset:010} 00000 n \n".encode("utf-8"))

    trailer = (
        f"trailer\n<< /Size {len(offsets) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("utf-8")
    pdf_parts.append(trailer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"".join(pdf_parts))


def write_report_files(report_text: str, appendix_text: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "final_report.md").write_text(report_text, encoding="utf-8")
    (output_dir / "appendix.md").write_text(appendix_text, encoding="utf-8")
    pages = chunk_for_pdf(report_text)
    write_simple_pdf(pages, output_dir / "final_report.pdf")


def generate_report(
    results_root: Path, builder: Optional[ReportBuilder] = None
) -> Tuple[List[ScenarioAggregate], ReportArtifacts]:
    active_builder = builder or ReportBuilder()
    aggregates = load_aggregates(results_root)
    artifacts = active_builder.build(aggregates)
    return aggregates, artifacts


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate research report and appendix from aggregated runs."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing aggregates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write report artefacts.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        help="Optional path for the report log file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List aggregates without generating report files.",
    )
    add_format_flags(parser, default="text")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    finalize_format_args(args, remove_in="0.2.0")
    command = "leadlag-report"
    if argv:
        command = "leadlag-report " + " ".join(argv)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_path or output_dir / "generate_report.log"
    setup_logging(
        log_path,
        level=str(args.log_level).upper(),
        context={"module": "generate_report"},
    )
    logger = get_logger(
        "reporting.generate_report",
        context={"results_root": str(args.results_root.resolve()), "output_dir": str(output_dir)},
    )

    base_data = {
        "results_root": str(args.results_root.resolve()),
        "output_dir": str(output_dir),
    }

    if args.dry_run:
        dirs = discover_aggregate_dirs(args.results_root)
        logger.info("[dry-run] discovered aggregates", context={"count": len(dirs)})
        for path in dirs:
            logger.info("[dry-run] aggregate dir", context={"path": str(path)})
        dry_text = ["[dry-run] aggregate directories:"]
        if dirs:
            dry_text.extend(f"  - {path}" for path in dirs)
        else:
            dry_text.append("  (none found)")
        emit_formatted_output(
            args,
            data={**base_data, "aggregates": [str(path) for path in dirs], "dry_run": True},
            text="\n".join(dry_text),
            message="Report dry-run completed.",
            pretty=True,
            command=command,
        )
        return 0

    aggregates, artifacts = generate_report(args.results_root)
    aggregate_names = artifacts.aggregate_names

    output_dir.mkdir(parents=True, exist_ok=True)
    write_report_files(artifacts.report_markdown, artifacts.appendix_markdown, output_dir)

    generated_files = {
        "report_markdown": str(output_dir / "final_report.md"),
        "report_pdf": str(output_dir / "final_report.pdf"),
        "appendix_markdown": str(output_dir / "appendix.md"),
    }
    if aggregates:
        logger.info(
            "Report generated",
            context={"scenario_count": len(aggregates), "report_dir": str(output_dir)},
        )
    else:
        logger.warning("No aggregate directories found; generated empty report")

    success = bool(aggregate_names)
    errors = None
    if not success:
        errors = [{"code": "no_aggregates", "message": "No aggregate directories found."}]

    text_lines = [f"Report directory: {output_dir}"]
    if aggregate_names:
        text_lines.append("Included scenarios:")
        text_lines.extend(f"  - {name}" for name in aggregate_names)
    else:
        text_lines.append("No aggregates included in report.")

    message = "Report generated." if success else "Report generated without aggregates."

    emit_formatted_output(
        args,
        data={
            **base_data,
            "aggregates": aggregate_names,
            "metadata": artifacts.metadata,
            "generated_files": generated_files,
            "dry_run": False,
        },
        text="\n".join(text_lines),
        message=message,
        artifacts=generated_files,
        errors=errors,
        success=success,
        pretty=True,
        command=command,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
