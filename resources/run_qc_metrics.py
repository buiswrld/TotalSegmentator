#!/usr/bin/env python
"""
Compute QC metrics for a set of segmentation masks and print them as a table.

Usage:
    python resources/run_qc_metrics.py -m <mask_dir_or_multilabel.nii.gz> [-c ct.nii.gz] [-ta total] [-o out.json] [-report_csv out.csv]

Examples:
    # Directory of per-structure masks (e.g. a non --ml TotalSegmentator run), with CT for HU stats
    python resources/run_qc_metrics.py -m output/ -c ct.nii.gz

    # Single --ml multilabel output, task determines the label->name mapping
    python resources/run_qc_metrics.py -m output.nii.gz -ta total -c ct.nii.gz -o qc_metrics.json

    # Write every structure/metric to a CSV (e.g. to open in Excel or load with pandas)
    python resources/run_qc_metrics.py -m output/ -c ct.nii.gz -report_csv qc_metrics.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from totalsegmentator.qc import calculate_mask_metrics

# Curated subset shown in the table by default; pass --all for every computed metric.
DEFAULT_COLUMNS = [
    "num_voxels", "volume_mm3", "is_empty", "num_components",
    "largest_component_fraction", "touches_boundary", "boundary_fraction",
    "mask_to_bbox_ratio", "mean_HU",
]


def format_table(metrics: dict, columns: list) -> str:
    headers = ["structure"] + columns
    rows = []
    for name, m in metrics.items():
        row = [name] + [m.get(c) for c in columns]
        rows.append(row)

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:,.2f}" if abs(v) >= 1 else f"{v:.4g}"
        return str(v)

    str_rows = [[fmt(v) for v in row] for row in rows]
    widths = [max(len(headers[i]), *(len(r[i]) for r in str_rows)) if str_rows else len(headers[i])
              for i in range(len(headers))]

    def format_row(vals):
        return "  ".join(v.ljust(w) for v, w in zip(vals, widths))

    lines = [format_row(headers), "  ".join("-" * w for w in widths)]
    lines += [format_row(r) for r in str_rows]
    return "\n".join(lines)


def write_csv(metrics: dict, path: Path) -> None:
    """Write every structure/metric to a CSV, one row per structure, one column per metric."""
    columns = list(next(iter(metrics.values())))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["structure"] + columns)
        for name, m in metrics.items():
            writer.writerow([name] + [m[c] if m[c] is not None else "" for c in columns])


def main():
    parser = argparse.ArgumentParser(description="Compute per-structure QC metrics for segmentation masks.")
    parser.add_argument("-m", "--mask_path", required=True, type=Path,
                        help="Directory of per-structure mask .nii.gz files, or a single multilabel .nii.gz file.")
    parser.add_argument("-c", "--ct_path", type=Path, default=None,
                        help="Original CT the masks were derived from. Enables mean_HU/median_HU/std_HU/p05_HU/p95_HU. Optional.")
    parser.add_argument("-ta", "--task", default="total",
                        help="Task whose class_map to use for a multilabel mask_path (default: total). Ignored for directory input.")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Write the full metrics dict as JSON to this path, in addition to printing the table.")
    parser.add_argument("-report_csv", "--report_csv", type=Path, default=None,
                        help="Write every structure/metric as a clean CSV to this path, in addition to printing the table.")
    parser.add_argument("--all", action="store_true",
                        help="Print every computed metric column instead of the curated default subset.")
    parser.add_argument("--sort-by", default=None, metavar="METRIC",
                        help="Sort rows by this metric, descending (e.g. --sort-by volume_mm3). Default: input order.")
    args = parser.parse_args()

    if not args.mask_path.exists():
        parser.error(f"mask_path does not exist: {args.mask_path}")

    metrics = calculate_mask_metrics(args.mask_path, ct_path=args.ct_path, task=args.task)

    if not metrics:
        print("No structures found.", file=sys.stderr)
        sys.exit(1)

    if args.sort_by is not None:
        first = next(iter(metrics.values()))
        if args.sort_by not in first:
            parser.error(f"unknown metric '{args.sort_by}'. Available: {', '.join(first)}")
        metrics = dict(sorted(metrics.items(), key=lambda kv: (kv[1][args.sort_by] is None, kv[1][args.sort_by]), reverse=True))

    columns = list(next(iter(metrics.values()))) if args.all else DEFAULT_COLUMNS
    print(format_table(metrics, columns))

    n_empty = sum(m["is_empty"] for m in metrics.values())
    print(f"\n{len(metrics)} structures, {n_empty} empty" + (" (no CT given: HU metrics skipped)" if args.ct_path is None else ""))

    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Full metrics written to {args.output}")

    if args.report_csv is not None:
        write_csv(metrics, args.report_csv)
        print(f"CSV report written to {args.report_csv}")


if __name__ == "__main__":
    main()
