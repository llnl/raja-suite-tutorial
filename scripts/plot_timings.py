#!/usr/bin/env python3
"""Plot a timing table and report the fastest variant.

Usage:
    command-that-produces-the-table | python3 plot_timings.py
    python3 plot_timings.py timing-output.txt
"""

import argparse
import re
import sys


ROW = re.compile(
    r"^(?P<name>\S+)\s+"
    r"(?P<elapsed>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<inclusive>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<elapsed_pct>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<inclusive_pct>[0-9]+(?:\.[0-9]+)?)\s*$"
)


def read_rows(stream):
    rows = []
    for line in stream:
        match = ROW.match(line.strip())
        if match:
            row = match.groupdict()
            rows.append((row["name"], float(row["elapsed"])))
    if not rows:
        raise ValueError("No timing rows found in input")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", help="timing output file; defaults to stdin")
    parser.add_argument("-o", "--output", help="save the plot instead of displaying it")
    parser.add_argument(
        "--all", action="store_true", help="include setup and non-multiply rows"
    )
    args = parser.parse_args()

    if args.input:
        with open(args.input, encoding="utf-8") as stream:
            rows = read_rows(stream)
    else:
        rows = read_rows(sys.stdin)

    if not args.all:
        rows = [row for row in rows if row[0].startswith("matrix_multiply")]
        if not rows:
            raise ValueError("No matrix_multiply variants found; rerun with --all")

    fastest_name, fastest_time = min(rows, key=lambda row: row[1])
    print(f"Fastest: {fastest_name} ({fastest_time:.6f} s)")

    import matplotlib.pyplot as plt

    names, times = zip(*rows)
    colors = ["tab:green" if name == fastest_name else "tab:blue" for name in names]
    figure_width = max(9, len(names) * 0.8)
    plt.figure(figsize=(figure_width, 5))
    bars = plt.bar(range(len(names)), times, color=colors)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.ylabel("Elapsed time (s)")
    plt.title("Timing variants (green = fastest)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
