"""Build a consolidated HTML report for paper artefacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

STYLE = """
body { font-family: "Segoe UI", Roboto, sans-serif; margin: 2rem; color: #0a0a0a; }
h1 { font-size: 1.8rem; margin-bottom: 1rem; }
h2 { font-size: 1.4rem; margin-top: 1.5rem; margin-bottom: 0.75rem; }
table { border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }
th, td { border: 1px solid #d0d7de; padding: 0.4rem 0.6rem; text-align: right; }
th { background: #f6f8fa; text-align: center; }
td:first-child, th:first-child { text-align: left; }
figure { margin: 1.5rem 0; }
figure img { max-width: 100%; height: auto; border: 1px solid #d0d7de; }
figure figcaption { font-size: 0.9rem; color: #57606a; margin-top: 0.4rem; }
"""

CSV_SECTIONS: Sequence[str] = (
    "main_results.csv",
    "ablations.csv",
    "hac_sharpe_ci.csv",
    "psr_dsr_pvalues.csv",
    "spa_table.csv",
    "mcs_table.csv",
)

FIGURE_FILES: Sequence[str] = (
    "forest.png",
    "heatmap.png",
    "pnl.png",
)


def _table_html(path: Path) -> str:
    frame = pd.read_csv(path)
    return frame.to_html(index=False, escape=False, border=0)


def _render_csv_sections(out_dir: Path) -> Iterable[str]:
    for filename in CSV_SECTIONS:
        path = out_dir / filename
        if not path.exists():
            continue
        yield f"<h2>{filename}</h2>\n" + _table_html(path)


def _render_figures(out_dir: Path) -> Iterable[str]:
    for filename in FIGURE_FILES:
        path = out_dir / filename
        if not path.exists():
            continue
        yield (
            "<figure>\n"
            f"  <img src='{filename}' alt='{filename}' />\n"
            f"  <figcaption>{filename}</figcaption>\n"
            "</figure>"
        )


def build(out_dir: str | Path, *, title: str = "Paper Report") -> Path:
    output_dir = Path(out_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory {output_dir} does not exist")

    sections = ["<h1>{}</h1>".format(title)]
    sections.extend(_render_csv_sections(output_dir))
    sections.extend(_render_figures(output_dir))

    html = (
        "<html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        f"<style>{STYLE}</style>"
        "</head><body>"
        + "".join(sections)
        + "</body></html>"
    )

    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory containing paper artefacts.",
    )
    parser.add_argument(
        "--title",
        default="Paper Report",
        help="Page title to embed in the HTML document.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    build(args.out, title=args.title)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
