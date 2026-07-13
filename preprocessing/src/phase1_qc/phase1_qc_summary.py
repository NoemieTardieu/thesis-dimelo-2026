import argparse
import csv
import glob
import math
import os
import struct
import zipfile
from ast import literal_eval
from typing import Dict, List, Tuple


def percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = int(q * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def pearson(xs: List[float], ys: List[float]) -> float:
    n = min(len(xs), len(ys))
    if n == 0:
        return float("nan")
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs[:n], ys[:n]))
    vx = sum((x - mx) ** 2 for x in xs[:n])
    vy = sum((y - my) ** 2 for y in ys[:n])
    if vx <= 0 or vy <= 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def read_tsv_summary(tsv_path: str) -> Dict[str, float]:
    mapq_vals: List[float] = []
    rl_vals: List[float] = []
    a_den_vals: List[float] = []
    c_den_vals: List[float] = []
    zero_a = 0
    zero_c = 0

    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mapq = float(row["mapq"])
            read_len = float(row["read_len"])
            a_den = float(row["A_mod_density_per_kb"])
            c_den = float(row["C_mod_density_per_kb"])
            a_conf = float(row["A_mod_conf"])
            c_conf = float(row["C_mod_conf"])

            mapq_vals.append(mapq)
            rl_vals.append(read_len)
            a_den_vals.append(a_den)
            c_den_vals.append(c_den)
            if a_conf == 0:
                zero_a += 1
            if c_conf == 0:
                zero_c += 1

    n = len(mapq_vals)
    mapq_s = sorted(mapq_vals)
    rl_s = sorted(rl_vals)
    a_s = sorted(a_den_vals)
    c_s = sorted(c_den_vals)

    return {
        "n_reads": n,
        "mapq_min": mapq_s[0] if n else float("nan"),
        "mapq_p50": percentile(mapq_s, 0.50),
        "mapq_p95": percentile(mapq_s, 0.95),
        "mapq_max": mapq_s[-1] if n else float("nan"),
        "mapq_mean": sum(mapq_vals) / n if n else float("nan"),
        "read_len_p5": percentile(rl_s, 0.05),
        "read_len_p50": percentile(rl_s, 0.50),
        "read_len_p95": percentile(rl_s, 0.95),
        "read_len_mean": (sum(rl_vals) / n if n else float("nan")),
        "a_density_p50": percentile(a_s, 0.50),
        "a_density_p95": percentile(a_s, 0.95),
        "a_density_mean": (sum(a_den_vals) / n if n else float("nan")),
        "c_density_p50": percentile(c_s, 0.50),
        "c_density_p95": percentile(c_s, 0.95),
        "c_density_mean": (sum(c_den_vals) / n if n else float("nan")),
        "a_zero_pct": (100.0 * zero_a / n if n else float("nan")),
        "c_zero_pct": (100.0 * zero_c / n if n else float("nan")),
        "corr_readlen_a_density": pearson(rl_vals, a_den_vals),
        "corr_readlen_c_density": pearson(rl_vals, c_den_vals),
    }


def read_tsv_columns(tsv_path: str) -> Tuple[List[float], List[float], List[float]]:
    read_len_vals: List[float] = []
    a_den_vals: List[float] = []
    c_den_vals: List[float] = []
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            read_len_vals.append(float(row["read_len"]))
            a_den_vals.append(float(row["A_mod_density_per_kb"]))
            c_den_vals.append(float(row["C_mod_density_per_kb"]))
    return read_len_vals, a_den_vals, c_den_vals


def parse_npz_shapes(npz_path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with zipfile.ZipFile(npz_path, "r") as zf:
        for name in sorted(zf.namelist()):
            data = zf.read(name)
            if data[:6] != b"\x93NUMPY":
                out[name] = "not_npy"
                continue
            major = data[6]
            if major == 1:
                hlen = struct.unpack("<H", data[8:10])[0]
                start = 10
            else:
                hlen = struct.unpack("<I", data[8:12])[0]
                start = 12
            header = data[start : start + hlen].decode("latin1").strip()
            parsed = literal_eval(header)
            out[name] = f"dtype={parsed.get('descr')} shape={parsed.get('shape')}"
    return out


def write_tsv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def histogram(values: List[float], bins: int, xmin: float, xmax: float) -> List[int]:
    counts = [0 for _ in range(bins)]
    if not values or xmax <= xmin:
        return counts
    width = (xmax - xmin) / bins
    for v in values:
        if v < xmin:
            idx = 0
        elif v >= xmax:
            idx = bins - 1
        else:
            idx = int((v - xmin) / width)
            idx = max(0, min(idx, bins - 1))
        counts[idx] += 1
    return counts


def svg_grouped_bars(
    out_path: str,
    title: str,
    categories: List[str],
    series: List[Tuple[str, List[float], str]],
    y_label: str,
) -> None:
    width, height = 920, 480
    ml, mr, mt, mb = 80, 30, 60, 90
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    max_v = 0.0
    for _name, vals, _color in series:
        if vals:
            max_v = max(max_v, max(vals))
    max_v = max_v * 1.1 if max_v > 0 else 1.0

    lines: List[str] = []
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
        y = mt + plot_h - (plot_h * t / 5)
        lines.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+plot_w}" y2="{y:.1f}" stroke="#ececec"/>')
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
            v = vals[i] if i < len(vals) else 0.0
            bh = (v / max_v) * plot_h
            x = gx + j * bar_w + (group_w - n_series * bar_w) / 2
            y = mt + plot_h - bh
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.8:.1f}" height="{bh:.1f}" fill="{color}"/>')
        lines.append(
            f'<text x="{gx+group_w/2:.1f}" y="{mt+plot_h+20}" text-anchor="middle" font-size="12" font-family="Arial">{cat}</text>'
        )

    legend_x = ml + 10
    legend_y = 40
    for i, (name, _vals, color) in enumerate(series):
        y = legend_y + i * 20
        lines.append(f'<rect x="{legend_x}" y="{y-10}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{legend_x+18}" y="{y}" font-size="12" font-family="Arial">{name}</text>')

    lines.append(
        f'<text x="{ml-55}" y="{mt+plot_h/2:.1f}" text-anchor="middle" transform="rotate(-90 {ml-55},{mt+plot_h/2:.1f})" font-size="12" font-family="Arial">{y_label}</text>'
    )
    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def svg_hist_overlay(
    out_path: str,
    title: str,
    x_label: str,
    series: List[Tuple[str, List[float], str]],
    bins: int = 50,
) -> None:
    width, height = 920, 480
    ml, mr, mt, mb = 80, 30, 60, 90
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    all_vals = []
    for _name, vals, _color in series:
        all_vals.extend(vals)
    xmin = min(all_vals) if all_vals else 0.0
    xmax = max(all_vals) if all_vals else 1.0
    if xmin == xmax:
        xmax = xmin + 1.0

    hists: List[Tuple[str, List[int], str]] = []
    max_count = 1
    for name, vals, color in series:
        h = histogram(vals, bins=bins, xmin=xmin, xmax=xmax)
        max_count = max(max_count, max(h) if h else 1)
        hists.append((name, h, color))

    lines: List[str] = []
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
        y = mt + plot_h - (plot_h * t / 5)
        lines.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+plot_w}" y2="{y:.1f}" stroke="#ececec"/>')
        lines.append(
            f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{int(yv)}</text>'
        )

    for name, h, color in hists:
        points = []
        for i, c in enumerate(h):
            x = ml + (i / (bins - 1)) * plot_w if bins > 1 else ml
            y = mt + plot_h - (c / max_count) * plot_h
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/>'
        )

    legend_x = ml + 10
    legend_y = 40
    for i, (name, _vals, color) in enumerate(series):
        y = legend_y + i * 20
        lines.append(f'<line x1="{legend_x}" y1="{y-4}" x2="{legend_x+12}" y2="{y-4}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x+18}" y="{y}" font-size="12" font-family="Arial">{name}</text>')

    lines.append(
        f'<text x="{ml+plot_w/2:.1f}" y="{height-30}" text-anchor="middle" font-size="12" font-family="Arial">{x_label}</text>'
    )
    lines.append(
        f'<text x="{ml-55}" y="{mt+plot_h/2:.1f}" text-anchor="middle" transform="rotate(-90 {ml-55},{mt+plot_h/2:.1f})" font-size="12" font-family="Arial">Count</text>'
    )
    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_markdown_report(
    report_path: str,
    summary_rows: List[Dict[str, object]],
    npz_rows: List[Dict[str, object]],
) -> None:
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 1 QC Summary\n\n")
        f.write("## Per-mark headline metrics\n\n")
        f.write("| Mark | Reads | MAPQ mean | ReadLen p50 | A dens mean | C dens mean | A zero % | C zero % |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                "| {mark} | {n_reads} | {mapq_mean:.2f} | {read_len_p50:.0f} | {a_density_mean:.3f} | {c_density_mean:.3f} | {a_zero_pct:.2f} | {c_zero_pct:.2f} |\n".format(
                    **row
                )
            )
        f.write("\n## NPZ content\n\n")
        f.write("| Mark | File | A tracks | C tracks | bin_size | ml_threshold |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in npz_rows:
            f.write(
                f"| {row['mark']} | {row['file']} | {row['a_tracks_shape']} | {row['c_tracks_shape']} | {row['bin_size_shape']} | {row['ml_threshold_shape']} |\n"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate summary tables and visualizations for Phase 1 QC outputs.")
    p.add_argument(
        "--input-dir",
        default="thesis_dimelo/src/preprocessing/phase1_output",
        help="Directory containing *.per_read.tsv and *.binned_tracks.npz files.",
    )
    p.add_argument(
        "--output-dir",
        default="thesis_dimelo/src/preprocessing/phase1_output/summary_report",
        help="Directory where summary tables/figures will be written.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tsv_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.per_read.tsv")))
    if not tsv_paths:
        raise SystemExit(f"No .per_read.tsv files found in: {args.input_dir}")

    summary_rows: List[Dict[str, object]] = []
    per_mark_columns: Dict[str, Tuple[List[float], List[float], List[float]]] = {}

    for tsv_path in tsv_paths:
        mark = os.path.basename(tsv_path).replace(".per_read.tsv", "")
        row = {"mark": mark}
        row.update(read_tsv_summary(tsv_path))
        summary_rows.append(row)
        per_mark_columns[mark] = read_tsv_columns(tsv_path)

    summary_fields = [
        "mark",
        "n_reads",
        "mapq_min",
        "mapq_p50",
        "mapq_p95",
        "mapq_max",
        "mapq_mean",
        "read_len_p5",
        "read_len_p50",
        "read_len_p95",
        "read_len_mean",
        "a_density_p50",
        "a_density_p95",
        "a_density_mean",
        "c_density_p50",
        "c_density_p95",
        "c_density_mean",
        "a_zero_pct",
        "c_zero_pct",
        "corr_readlen_a_density",
        "corr_readlen_c_density",
    ]
    write_tsv(os.path.join(args.output_dir, "phase1_qc_summary.tsv"), summary_rows, summary_fields)

    npz_rows: List[Dict[str, object]] = []
    for npz_path in sorted(glob.glob(os.path.join(args.input_dir, "*.binned_tracks.npz"))):
        mark = os.path.basename(npz_path).replace(".binned_tracks.npz", "")
        shapes = parse_npz_shapes(npz_path)
        npz_rows.append(
            {
                "mark": mark,
                "file": os.path.basename(npz_path),
                "a_tracks_shape": shapes.get("A_tracks.npy", "missing"),
                "c_tracks_shape": shapes.get("C_tracks.npy", "missing"),
                "bin_size_shape": shapes.get("bin_size.npy", "missing"),
                "ml_threshold_shape": shapes.get("ml_threshold.npy", "missing"),
            }
        )
    write_tsv(
        os.path.join(args.output_dir, "phase1_npz_inventory.tsv"),
        npz_rows,
        ["mark", "file", "a_tracks_shape", "c_tracks_shape", "bin_size_shape", "ml_threshold_shape"],
    )

    marks = [row["mark"] for row in summary_rows]
    svg_grouped_bars(
        out_path=os.path.join(args.output_dir, "figure_mean_density_by_mark.svg"),
        title="Mean Modification Density per kb by Mark",
        categories=marks,
        series=[
            ("A-mod mean density", [float(r["a_density_mean"]) for r in summary_rows], "#2c7fb8"),
            ("C-mod mean density", [float(r["c_density_mean"]) for r in summary_rows], "#d95f0e"),
        ],
        y_label="mods per kb",
    )
    svg_grouped_bars(
        out_path=os.path.join(args.output_dir, "figure_zero_conf_fraction_by_mark.svg"),
        title="Fraction of Reads with Zero Confident Mods",
        categories=marks,
        series=[
            ("A_mod_conf == 0 (%)", [float(r["a_zero_pct"]) for r in summary_rows], "#4daf4a"),
            ("C_mod_conf == 0 (%)", [float(r["c_zero_pct"]) for r in summary_rows], "#984ea3"),
        ],
        y_label="percentage of reads",
    )
    svg_grouped_bars(
        out_path=os.path.join(args.output_dir, "figure_read_length_quantiles_by_mark.svg"),
        title="Read Length Quantiles by Mark",
        categories=marks,
        series=[
            ("ReadLen p50", [float(r["read_len_p50"]) for r in summary_rows], "#377eb8"),
            ("ReadLen p95", [float(r["read_len_p95"]) for r in summary_rows], "#e41a1c"),
        ],
        y_label="bp",
    )

    for mark, (read_len_vals, a_den_vals, c_den_vals) in per_mark_columns.items():
        svg_hist_overlay(
            out_path=os.path.join(args.output_dir, f"figure_{mark}_density_hist_overlay.svg"),
            title=f"{mark} Density Distribution (A vs C)",
            x_label="mods per kb",
            series=[
                ("A_mod_density_per_kb", a_den_vals, "#2c7fb8"),
                ("C_mod_density_per_kb", c_den_vals, "#d95f0e"),
            ],
            bins=60,
        )
        svg_hist_overlay(
            out_path=os.path.join(args.output_dir, f"figure_{mark}_read_length_hist.svg"),
            title=f"{mark} Read Length Distribution",
            x_label="read length (bp)",
            series=[("read_len", read_len_vals, "#1b9e77")],
            bins=60,
        )

    write_markdown_report(
        os.path.join(args.output_dir, "phase1_qc_report.md"),
        summary_rows=summary_rows,
        npz_rows=npz_rows,
    )

    print(f"Wrote summary artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
