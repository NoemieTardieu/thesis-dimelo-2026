#!/usr/bin/env python3
"""Analyze 5mC/6mA relationship on final HyenaDNA tensor datasets.

This script works on the same npz + metadata tensors used for HyenaDNA
training/evaluation. It reports M-vs-Reg relationships at three units:

1. window: one point per tensor row/window
2. read_overlap_collapsed: one point per sample/read_id after collapsing
   duplicate observations from overlapping windows by read position
3. region: one point per genomic region, averaging all windows/reads

5mC and 6mA are observed at different base types, so the correlation is between
aggregate signal summaries, not exact same-base observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    npz: Path
    metadata: Path
    label: str


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("rt", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def finite_pair_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(y)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks, 1-based, implemented with numpy only."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def corr_stats(x_values: Iterable[float], y_values: Iterable[float]) -> dict[str, float | int]:
    x = np.asarray(list(x_values), dtype=np.float64)
    y = np.asarray(list(y_values), dtype=np.float64)
    mask = finite_pair_mask(x, y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    out: dict[str, float | int] = {"n": n}
    if n < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        out.update({"pearson": float("nan"), "spearman": float("nan")})
        return out
    out["pearson"] = float(np.corrcoef(x, y)[0, 1])
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    out["spearman"] = float(np.corrcoef(rx, ry)[0, 1])
    return out


def mean_or_nan(values: np.ndarray, mask: np.ndarray) -> float:
    if not bool(mask.any()):
        return float("nan")
    return float(values[mask].mean())


def window_rows(spec: DatasetSpec) -> list[dict[str, object]]:
    data = np.load(spec.npz)
    metadata = read_metadata(spec.metadata)
    if len(metadata) != data["target_5mC"].shape[0]:
        raise ValueError(f"metadata rows do not match tensor rows for {spec.npz}")

    rows: list[dict[str, object]] = []
    target_5mc = data["target_5mC"]
    mask_5mc = data["mask_5mC"].astype(bool)
    target_6ma = data["target_6mA"]
    mask_6ma = data["mask_6mA"].astype(bool)

    for idx, meta in enumerate(metadata):
        rows.append(
            {
                "dataset": spec.label,
                "split": meta.get("split", spec.label),
                "sample": meta.get("sample", ""),
                "chrom": meta.get("chrom", ""),
                "region_name": meta.get("region_name", meta.get("region_id", "")),
                "read_id": meta.get("read_id", ""),
                "m_mean": mean_or_nan(target_5mc[idx], mask_5mc[idx]),
                "reg_mean": mean_or_nan(target_6ma[idx], mask_6ma[idx]),
                "m_n": int(mask_5mc[idx].sum()),
                "reg_n": int(mask_6ma[idx].sum()),
            }
        )
    return rows


def read_overlap_collapsed_rows(spec: DatasetSpec) -> list[dict[str, object]]:
    data = np.load(spec.npz)
    metadata = read_metadata(spec.metadata)
    target_5mc = data["target_5mC"]
    mask_5mc = data["mask_5mC"].astype(bool)
    target_6ma = data["target_6mA"]
    mask_6ma = data["mask_6mA"].astype(bool)

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, meta in enumerate(metadata):
        grouped[(meta.get("sample", ""), meta.get("read_id", ""))].append(idx)

    rows: list[dict[str, object]] = []
    for (sample, read_id), indices in grouped.items():
        first = metadata[indices[0]]
        m_by_pos: dict[int, list[float]] = defaultdict(list)
        reg_by_pos: dict[int, list[float]] = defaultdict(list)
        for idx in indices:
            window_start = int(metadata[idx].get("window_start", "0"))

            m_pos = np.flatnonzero(mask_5mc[idx])
            for pos in m_pos:
                m_by_pos[window_start + int(pos)].append(float(target_5mc[idx, pos]))

            reg_pos = np.flatnonzero(mask_6ma[idx])
            for pos in reg_pos:
                reg_by_pos[window_start + int(pos)].append(float(target_6ma[idx, pos]))

        m_values = [float(np.mean(v)) for v in m_by_pos.values()]
        reg_values = [float(np.mean(v)) for v in reg_by_pos.values()]
        rows.append(
            {
                "dataset": spec.label,
                "split": first.get("split", spec.label),
                "sample": sample,
                "chrom": first.get("chrom", ""),
                "region_name": first.get("region_name", first.get("region_id", "")),
                "read_id": read_id,
                "m_mean": float(np.mean(m_values)) if m_values else float("nan"),
                "reg_mean": float(np.mean(reg_values)) if reg_values else float("nan"),
                "m_n": len(m_values),
                "reg_n": len(reg_values),
            }
        )
    return rows


def region_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(
        lambda: {"m_sum": 0.0, "m_n": 0.0, "reg_sum": 0.0, "reg_n": 0.0}
    )
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["sample"]),
            str(row["chrom"]),
            str(row["region_name"]),
        )
        if math.isfinite(float(row["m_mean"])) and int(row["m_n"]) > 0:
            grouped[key]["m_sum"] += float(row["m_mean"]) * int(row["m_n"])
            grouped[key]["m_n"] += int(row["m_n"])
        if math.isfinite(float(row["reg_mean"])) and int(row["reg_n"]) > 0:
            grouped[key]["reg_sum"] += float(row["reg_mean"]) * int(row["reg_n"])
            grouped[key]["reg_n"] += int(row["reg_n"])

    out: list[dict[str, object]] = []
    for (dataset, sample, chrom, region_name), vals in grouped.items():
        out.append(
            {
                "dataset": dataset,
                "split": dataset,
                "sample": sample,
                "chrom": chrom,
                "region_name": region_name,
                "read_id": "",
                "m_mean": vals["m_sum"] / vals["m_n"] if vals["m_n"] else float("nan"),
                "reg_mean": vals["reg_sum"] / vals["reg_n"] if vals["reg_n"] else float("nan"),
                "m_n": int(vals["m_n"]),
                "reg_n": int(vals["reg_n"]),
            }
        )
    return out


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "dataset",
        "split",
        "sample",
        "chrom",
        "region_name",
        "read_id",
        "m_mean",
        "reg_mean",
        "m_n",
        "reg_n",
    ]
    with path.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_correlation_summary(path: Path, unit_rows: dict[str, list[dict[str, object]]]) -> None:
    with path.open("wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["unit", "scope", "group", "n", "pearson", "spearman"],
        )
        writer.writeheader()
        for unit, rows in unit_rows.items():
            scopes: list[tuple[str, str, list[dict[str, object]]]] = [("all", "all", rows)]
            for sample in sorted({str(r["sample"]) for r in rows if str(r["sample"])}):
                scopes.append(("sample", sample, [r for r in rows if str(r["sample"]) == sample]))
            for chrom in sorted({str(r["chrom"]) for r in rows if str(r["chrom"])}):
                scopes.append(("chrom", chrom, [r for r in rows if str(r["chrom"]) == chrom]))
            for scope, group, subset in scopes:
                stats = corr_stats((float(r["m_mean"]) for r in subset), (float(r["reg_mean"]) for r in subset))
                writer.writerow(
                    {
                        "unit": unit,
                        "scope": scope,
                        "group": group,
                        "n": stats["n"],
                        "pearson": stats["pearson"],
                        "spearman": stats["spearman"],
                    }
                )


def write_cooccurrence(path: Path, rows: list[dict[str, object]], n_bins: int) -> None:
    m = np.asarray([float(r["m_mean"]) for r in rows], dtype=np.float64)
    reg = np.asarray([float(r["reg_mean"]) for r in rows], dtype=np.float64)
    mask = finite_pair_mask(m, reg)
    m = m[mask]
    reg = reg[mask]
    if m.size < n_bins * n_bins:
        raise ValueError("not enough valid paired rows for cooccurrence")
    m_edges = np.quantile(m, np.linspace(0.0, 1.0, n_bins + 1))
    reg_edges = np.quantile(reg, np.linspace(0.0, 1.0, n_bins + 1))
    m_bin = np.clip(np.searchsorted(m_edges, m, side="right") - 1, 0, n_bins - 1)
    reg_bin = np.clip(np.searchsorted(reg_edges, reg, side="right") - 1, 0, n_bins - 1)
    counts = np.zeros((n_bins, n_bins), dtype=np.int64)
    for i, j in zip(m_bin, reg_bin):
        counts[int(i), int(j)] += 1
    expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / counts.sum()
    enrichment = np.divide(counts, expected, out=np.full_like(expected, np.nan, dtype=np.float64), where=expected > 0)

    with path.open("wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=[
                "m_bin",
                "reg_bin",
                "count",
                "expected_if_independent",
                "enrichment",
                "m_low",
                "m_high",
                "reg_low",
                "reg_high",
            ],
        )
        writer.writeheader()
        for i in range(n_bins):
            for j in range(n_bins):
                writer.writerow(
                    {
                        "m_bin": i + 1,
                        "reg_bin": j + 1,
                        "count": int(counts[i, j]),
                        "expected_if_independent": float(expected[i, j]),
                        "enrichment": float(enrichment[i, j]),
                        "m_low": float(m_edges[i]),
                        "m_high": float(m_edges[i + 1]),
                        "reg_low": float(reg_edges[j]),
                        "reg_high": float(reg_edges[j + 1]),
                    }
                )


def make_plot(path: Path, unit_rows: dict[str, list[dict[str, object]]], n_bins: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(unit_rows), figsize=(6 * len(unit_rows), 5), constrained_layout=True)
    if len(unit_rows) == 1:
        axes = [axes]
    for ax, (unit, rows) in zip(axes, unit_rows.items()):
        m = np.asarray([float(r["m_mean"]) for r in rows], dtype=np.float64)
        reg = np.asarray([float(r["reg_mean"]) for r in rows], dtype=np.float64)
        mask = finite_pair_mask(m, reg)
        m = m[mask]
        reg = reg[mask]
        h = ax.hexbin(m, reg, gridsize=60, mincnt=1, bins="log", cmap="viridis")
        stats = corr_stats(m, reg)
        ax.set_title(f"{unit}\nn={stats['n']}, rho={stats['spearman']:.3f}")
        ax.set_xlabel("mean 5mC / M")
        ax.set_ylabel("mean 6mA / Reg")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.colorbar(h, ax=ax, label="log10(count)")
    fig.suptitle("M-Reg relationship on final HyenaDNA tensors")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=3,
        metavar=("NPZ", "METADATA", "LABEL"),
        required=True,
        help="May be repeated. Label should encode chrom/split, e.g. chr16_test.",
    )
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--cooccurrence-bins", type=int, default=5)
    parser.add_argument("--skip-read-overlap-collapse", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [DatasetSpec(Path(npz), Path(metadata), label) for npz, metadata, label in args.dataset]
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    all_window_rows: list[dict[str, object]] = []
    all_read_rows: list[dict[str, object]] = []
    for spec in specs:
        print(json.dumps({"progress": "window_rows", "dataset": spec.label}), flush=True)
        all_window_rows.extend(window_rows(spec))
        if not args.skip_read_overlap_collapse:
            print(json.dumps({"progress": "read_overlap_collapsed_rows", "dataset": spec.label}), flush=True)
            all_read_rows.extend(read_overlap_collapsed_rows(spec))

    all_region_rows = region_rows(all_read_rows if all_read_rows else all_window_rows)
    unit_rows = {"window": all_window_rows, "region": all_region_rows}
    if all_read_rows:
        unit_rows = {"window": all_window_rows, "read_overlap_collapsed": all_read_rows, "region": all_region_rows}

    for unit, rows in unit_rows.items():
        write_rows(out_prefix.with_suffix(f".{unit}.tsv"), rows)

    write_correlation_summary(out_prefix.with_suffix(".correlation_summary.tsv"), unit_rows)
    if all_read_rows:
        write_cooccurrence(out_prefix.with_suffix(".read_overlap_collapsed.cooccurrence.tsv"), all_read_rows, args.cooccurrence_bins)
    write_cooccurrence(out_prefix.with_suffix(".window.cooccurrence.tsv"), all_window_rows, args.cooccurrence_bins)
    plot_path = out_prefix.with_suffix(".hexbin.png")
    try:
        make_plot(plot_path, unit_rows, args.cooccurrence_bins)
    except ModuleNotFoundError as exc:
        print(json.dumps({"warning": "plot_skipped_missing_module", "module": str(exc)}), flush=True)
        plot_path = Path("")

    summary = {
        "datasets": [{"npz": str(s.npz), "metadata": str(s.metadata), "label": s.label} for s in specs],
        "outputs": {
            "window": str(out_prefix.with_suffix(".window.tsv")),
            "read_overlap_collapsed": str(out_prefix.with_suffix(".read_overlap_collapsed.tsv")) if all_read_rows else None,
            "region": str(out_prefix.with_suffix(".region.tsv")),
            "correlation_summary": str(out_prefix.with_suffix(".correlation_summary.tsv")),
            "window_cooccurrence": str(out_prefix.with_suffix(".window.cooccurrence.tsv")),
            "read_overlap_collapsed_cooccurrence": str(out_prefix.with_suffix(".read_overlap_collapsed.cooccurrence.tsv")) if all_read_rows else None,
            "plot": str(plot_path) if str(plot_path) else None,
        },
    }
    with out_prefix.with_suffix(".summary.json").open("wt") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
