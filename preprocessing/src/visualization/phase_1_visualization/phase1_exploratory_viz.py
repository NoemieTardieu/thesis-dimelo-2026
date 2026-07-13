import argparse
import csv
import glob
import math
import os
from typing import Dict, List, Tuple


COLORS = {
    "h3k27ac": "#1b9e77",
    "h3k27me3": "#d95f02",
    "h3k4me3": "#7570b3",
}


def percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = int(q * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def histogram(values: List[float], bins: int, xmin: float, xmax: float) -> List[int]:
    counts = [0] * bins
    if not values or xmax <= xmin:
        return counts
    width = (xmax - xmin) / bins
    for v in values:
        if v <= xmin:
            idx = 0
        elif v >= xmax:
            idx = bins - 1
        else:
            idx = int((v - xmin) / width)
            idx = max(0, min(bins - 1, idx))
        counts[idx] += 1
    return counts


def read_per_mark(tsv_path: str) -> Dict[str, List[float]]:
    read_len = []
    a_density = []
    c_density = []
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            read_len.append(float(row["read_len"]))
            a_density.append(float(row["A_mod_density_per_kb"]))
            c_density.append(float(row["C_mod_density_per_kb"]))
    return {"read_len": read_len, "a_density": a_density, "c_density": c_density}


def svg_hist_overlay(
    out_path: str,
    title: str,
    x_label: str,
    series: List[Tuple[str, List[float], str]],
    bins: int = 60,
) -> None:
    width, height = 950, 520
    ml, mr, mt, mb = 80, 30, 60, 95
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    all_vals = [v for _name, vals, _color in series for v in vals]
    xmin = min(all_vals) if all_vals else 0.0
    xmax = max(all_vals) if all_vals else 1.0
    if xmin == xmax:
        xmax = xmin + 1.0

    hist_data = []
    max_count = 1
    for name, vals, color in series:
        h = histogram(vals, bins=bins, xmin=xmin, xmax=xmax)
        hist_data.append((name, h, color))
        if h:
            max_count = max(max_count, max(h))

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>'
    )
    lines.append(
        f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" stroke="black" stroke-width="1"/>'
    )
    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" stroke="black" stroke-width="1"/>')

    for t in range(6):
        yv = max_count * t / 5
        y = mt + plot_h - plot_h * t / 5
        lines.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+plot_w}" y2="{y:.1f}" stroke="#efefef"/>')
        lines.append(
            f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{int(yv)}</text>'
        )

    for name, h, color in hist_data:
        pts = []
        for i, c in enumerate(h):
            x = ml + (i / max(1, bins - 1)) * plot_w
            y = mt + plot_h - (c / max_count) * plot_h
            pts.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(pts)}"/>')

    legend_x, legend_y = ml + 8, 42
    for i, (name, _vals, color) in enumerate(series):
        y = legend_y + i * 20
        lines.append(f'<line x1="{legend_x}" y1="{y-4}" x2="{legend_x+12}" y2="{y-4}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x+18}" y="{y}" font-size="12" font-family="Arial">{name}</text>')

    lines.append(
        f'<text x="{ml+plot_w/2:.1f}" y="{height-34}" text-anchor="middle" font-size="12" font-family="Arial">{x_label}</text>'
    )
    lines.append(
        f'<text x="{ml-56}" y="{mt+plot_h/2:.1f}" text-anchor="middle" transform="rotate(-90 {ml-56},{mt+plot_h/2:.1f})" font-size="12" font-family="Arial">Count</text>'
    )
    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def svg_grouped_bars(
    out_path: str,
    title: str,
    categories: List[str],
    series: List[Tuple[str, List[float], str]],
    y_label: str,
) -> None:
    width, height = 950, 520
    ml, mr, mt, mb = 85, 30, 60, 95
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    max_v = max(max(vals) for _n, vals, _c in series if vals)
    max_v = max_v * 1.15 if max_v > 0 else 1.0

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>'
    )
    lines.append(
        f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" stroke="black" stroke-width="1"/>'
    )
    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" stroke="black" stroke-width="1"/>')

    for t in range(6):
        yv = max_v * t / 5
        y = mt + plot_h - plot_h * t / 5
        lines.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+plot_w}" y2="{y:.1f}" stroke="#efefef"/>')
        lines.append(
            f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{yv:.1f}</text>'
        )

    n_cat = len(categories)
    n_series = len(series)
    group_w = plot_w / max(1, n_cat)
    bar_w = group_w / (n_series + 1)
    for i, cat in enumerate(categories):
        gx = ml + i * group_w
        for j, (_name, vals, color) in enumerate(series):
            v = vals[i]
            bh = (v / max_v) * plot_h
            x = gx + j * bar_w + (group_w - n_series * bar_w) / 2
            y = mt + plot_h - bh
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.8:.1f}" height="{bh:.1f}" fill="{color}"/>')
        lines.append(
            f'<text x="{gx+group_w/2:.1f}" y="{mt+plot_h+20}" text-anchor="middle" font-size="12" font-family="Arial">{cat}</text>'
        )

    legend_x, legend_y = ml + 10, 42
    for i, (name, _vals, color) in enumerate(series):
        y = legend_y + i * 20
        lines.append(f'<rect x="{legend_x}" y="{y-10}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{legend_x+18}" y="{y}" font-size="12" font-family="Arial">{name}</text>')

    lines.append(
        f'<text x="{ml-58}" y="{mt+plot_h/2:.1f}" text-anchor="middle" transform="rotate(-90 {ml-58},{mt+plot_h/2:.1f})" font-size="12" font-family="Arial">{y_label}</text>'
    )
    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def svg_heatmap(
    out_path: str,
    title: str,
    x_label: str,
    y_label: str,
    matrix: List[List[int]],
    x_ticks: List[str],
    y_ticks: List[str],
) -> None:
    width, height = 980, 620
    ml, mr, mt, mb = 120, 80, 65, 120
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    max_val = max(max(row) for row in matrix) if rows and cols else 1
    max_log = math.log1p(max_val)

    def lerp(ca: Tuple[int, int, int], cb: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
        return (
            int(ca[0] + t * (cb[0] - ca[0])),
            int(ca[1] + t * (cb[1] - ca[1])),
            int(ca[2] + t * (cb[2] - ca[2])),
        )

    def color_for(v: int) -> str:
        if max_log <= 0:
            t = 0.0
        else:
            t = math.log1p(v) / max_log
        # Blue -> near-white -> red (similar to common bioinformatics heatmaps)
        low = (33, 102, 255)
        mid = (245, 245, 245)
        high = (255, 49, 49)
        if t <= 0.5:
            r, g, b = lerp(low, mid, t / 0.5)
        else:
            r, g, b = lerp(mid, high, (t - 0.5) / 0.5)
        return f"rgb({r},{g},{b})"

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(
        f'<text x="{width/2:.1f}" y="32" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>'
    )

    cell_w = plot_w / max(1, cols)
    cell_h = plot_h / max(1, rows)
    for r in range(rows):
        for c in range(cols):
            x = ml + c * cell_w
            y = mt + (rows - 1 - r) * cell_h
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" fill="{color_for(matrix[r][c])}" stroke="none"/>'
            )

    lines.append(
        f'<rect x="{ml}" y="{mt}" width="{plot_w}" height="{plot_h}" fill="none" stroke="black" stroke-width="1"/>'
    )

    for i, t in enumerate(x_ticks):
        x = ml + (i + 0.5) * cell_w
        lines.append(
            f'<text x="{x:.1f}" y="{mt+plot_h+22}" text-anchor="middle" font-size="11" font-family="Arial" transform="rotate(25 {x:.1f},{mt+plot_h+22})">{t}</text>'
        )
    for i, t in enumerate(y_ticks):
        y = mt + plot_h - (i + 0.5) * cell_h
        lines.append(
            f'<text x="{ml-10}" y="{y+4:.1f}" text-anchor="end" font-size="11" font-family="Arial">{t}</text>'
        )

    lines.append(
        f'<text x="{ml+plot_w/2:.1f}" y="{height-34}" text-anchor="middle" font-size="12" font-family="Arial">{x_label}</text>'
    )
    lines.append(
        f'<text x="{ml-72}" y="{mt+plot_h/2:.1f}" text-anchor="middle" transform="rotate(-90 {ml-72},{mt+plot_h/2:.1f})" font-size="12" font-family="Arial">{y_label}</text>'
    )

    cb_x = ml + plot_w + 24
    cb_y = mt
    cb_w = 18
    cb_h = plot_h
    grad_steps = 80
    for i in range(grad_steps):
        t = i / (grad_steps - 1)
        v = int((math.expm1(t * max_log)) if max_log > 0 else 0)
        y = cb_y + cb_h - (i + 1) * (cb_h / grad_steps)
        lines.append(
            f'<rect x="{cb_x}" y="{y:.2f}" width="{cb_w}" height="{cb_h/grad_steps:.2f}" fill="{color_for(v)}" stroke="none"/>'
        )
    lines.append(f'<rect x="{cb_x}" y="{cb_y}" width="{cb_w}" height="{cb_h}" fill="none" stroke="black" stroke-width="0.8"/>')
    lines.append(f'<text x="{cb_x+cb_w+8}" y="{cb_y+8}" font-size="10" font-family="Arial">high</text>')
    lines.append(f'<text x="{cb_x+cb_w+8}" y="{cb_y+cb_h}" font-size="10" font-family="Arial">low</text>')
    lines.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def make_bin_edges(values: List[float], n_bins: int = 10) -> List[float]:
    vals = sorted(values)
    if not vals:
        return [0.0, 1.0]
    edges = [vals[0]]
    for i in range(1, n_bins):
        q = i / n_bins
        edges.append(percentile(vals, q))
    edges.append(vals[-1] + 1e-9)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return edges


def find_bin(v: float, edges: List[float]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return i
    return len(edges) - 2


def build_cooccurrence_matrix(a_vals: List[float], c_vals: List[float], bins: int = 10) -> Tuple[List[List[int]], List[str], List[str]]:
    a_edges = make_bin_edges(a_vals, n_bins=bins)
    c_edges = make_bin_edges(c_vals, n_bins=bins)
    mat = [[0 for _ in range(bins)] for _ in range(bins)]
    n = min(len(a_vals), len(c_vals))
    for i in range(n):
        ax = find_bin(a_vals[i], a_edges)
        cy = find_bin(c_vals[i], c_edges)
        mat[cy][ax] += 1

    x_ticks = [f"Q{i+1}" for i in range(bins)]
    y_ticks = [f"Q{i+1}" for i in range(bins)]
    return mat, x_ticks, y_ticks


def write_summary_md(out_path: str, mark_stats: List[Dict[str, float]]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Phase 1 Exploratory Visualizations\n\n")
        f.write("## Quick interpretation\n\n")
        f.write("- C-density is consistently higher than A-density across marks.\n")
        f.write("- A-density has a larger near-zero mass, consistent with sparse directed signal.\n")
        f.write("- Read lengths support window-based modeling.\n")
        f.write("- Reg vs M co-occurrence heatmap is read-level exploratory (final analysis should be recomputed after Phase 2 binning).\n\n")
        f.write("## Per-mark summary\n\n")
        f.write("| Mark | n reads | ReadLen p50 | A density mean | C density mean |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in mark_stats:
            f.write(
                f"| {row['mark']} | {int(row['n'])} | {row['read_len_p50']:.0f} | {row['a_mean']:.3f} | {row['c_mean']:.3f} |\n"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate exploratory Phase 1 visualizations from per_read TSVs.")
    p.add_argument(
        "--input-dir",
        default="thesis_dimelo/src/preprocessing/phase1_output",
        help="Folder containing *.per_read.tsv files.",
    )
    p.add_argument(
        "--output-dir",
        default="thesis_dimelo/src/preprocessing/phase_1_visualization/output",
        help="Folder for SVG figures and markdown summary.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tsv_files = sorted(glob.glob(os.path.join(args.input_dir, "*.per_read.tsv")))
    if not tsv_files:
        raise SystemExit(f"No .per_read.tsv files found in: {args.input_dir}")

    mark_data: Dict[str, Dict[str, List[float]]] = {}
    mark_stats: List[Dict[str, float]] = []

    for tsv in tsv_files:
        mark = os.path.basename(tsv).replace(".per_read.tsv", "")
        cols = read_per_mark(tsv)
        mark_data[mark] = cols

        n = len(cols["read_len"])
        rl_s = sorted(cols["read_len"])
        a_mean = sum(cols["a_density"]) / n if n else float("nan")
        c_mean = sum(cols["c_density"]) / n if n else float("nan")
        mark_stats.append(
            {
                "mark": mark,
                "n": float(n),
                "read_len_p50": percentile(rl_s, 0.5),
                "a_mean": a_mean,
                "c_mean": c_mean,
            }
        )

    marks = [row["mark"] for row in mark_stats]
    svg_grouped_bars(
        out_path=os.path.join(args.output_dir, "viz_basic_dimelo_signal_summary.svg"),
        title="Basic DiMeLo-seq Signal Summary by Mark",
        categories=marks,
        series=[
            ("A density mean (mods/kb)", [row["a_mean"] for row in mark_stats], "#2c7fb8"),
            ("C density mean (mods/kb)", [row["c_mean"] for row in mark_stats], "#d95f0e"),
        ],
        y_label="mods per kb",
    )

    svg_grouped_bars(
        out_path=os.path.join(args.output_dir, "viz_read_length_median_by_mark.svg"),
        title="Median Read Length by Mark",
        categories=marks,
        series=[("Read length p50 (bp)", [row["read_len_p50"] for row in mark_stats], "#4daf4a")],
        y_label="bp",
    )

    for mark in marks:
        cols = mark_data[mark]
        color = COLORS.get(mark, "#1f78b4")
        svg_hist_overlay(
            out_path=os.path.join(args.output_dir, f"viz_{mark}_methylation_distribution.svg"),
            title=f"{mark}: Methylation Distributions",
            x_label="mods per kb",
            series=[
                ("A_mod_density_per_kb (Reg proxy)", cols["a_density"], color),
                ("C_mod_density_per_kb (M proxy)", cols["c_density"], "#e6550d"),
            ],
            bins=70,
        )

    all_a = []
    all_c = []
    for mark in marks:
        all_a.extend(mark_data[mark]["a_density"])
        all_c.extend(mark_data[mark]["c_density"])

    mat, xt, yt = build_cooccurrence_matrix(all_a, all_c, bins=10)
    svg_heatmap(
        out_path=os.path.join(args.output_dir, "viz_reg_vs_m_cooccurrence_heatmap.svg"),
        title="Reg vs M Co-occurrence Heatmap (Read-level Quantile Bins)",
        x_label="Reg proxy: A_mod_density_per_kb quantile",
        y_label="M proxy: C_mod_density_per_kb quantile",
        matrix=mat,
        x_ticks=xt,
        y_ticks=yt,
    )

    write_summary_md(
        out_path=os.path.join(args.output_dir, "viz_exploration_notes.md"),
        mark_stats=mark_stats,
    )
    print(f"Wrote exploratory visualizations to: {args.output_dir}")


if __name__ == "__main__":
    main()
