#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


KEYS = ["chrom", "region_id", "region_start", "region_end", "bin_start", "bin_end"]
ALPHA_DEFAULT = "A549_H3K4me3_fixed_mean"


def minmax(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    lo = values.min(skipna=True)
    hi = values.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return values * np.nan
    return (values - lo) / (hi - lo)


def basic_metrics(target: np.ndarray, prediction: np.ndarray, threshold: float | None) -> dict:
    mask = np.isfinite(target) & np.isfinite(prediction)
    target = target[mask]
    prediction = prediction[mask]
    result = {
        "number_of_bins": int(target.size),
        "pearson": np.nan,
        "spearman": np.nan,
        "auroc": np.nan,
        "auprc": np.nan,
        "positive_fraction": np.nan,
        "auprc_enrichment": np.nan,
    }
    if target.size < 2:
        return result
    if np.std(target) > 0 and np.std(prediction) > 0:
        result["pearson"] = float(pearsonr(target, prediction).statistic)
        result["spearman"] = float(spearmanr(target, prediction).statistic)
    if threshold is not None:
        labels = target >= threshold
        result["positive_fraction"] = float(labels.mean())
        if np.unique(labels).size == 2:
            result["auroc"] = float(roc_auc_score(labels, prediction))
            result["auprc"] = float(average_precision_score(labels, prediction))
            if result["positive_fraction"] > 0:
                result["auprc_enrichment"] = result["auprc"] / result["positive_fraction"]
    return result


def parse_bigwig_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for spec in specs:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            path = spec
            name = Path(path).name
            for suffix in (".bigWig", ".bw", ".bigwig"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break
        clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
        parsed.append((clean, Path(path)))
    return parsed


def sample_bigwigs(alpha: pd.DataFrame, specs: list[tuple[str, Path]]) -> pd.DataFrame:
    try:
        import pyBigWig
    except ImportError as exc:
        raise SystemExit(
            "pyBigWig is required to read bigWig files. Install it with:\n"
            "  /scratch/leuven/383/vsc38330/.venv/bin/python -m pip install pyBigWig"
        ) from exc

    rows = alpha[KEYS + ["region_name"]].drop_duplicates().copy()
    for name, path in specs:
        if not path.exists():
            raise SystemExit(f"bigWig not found: {path}")
        values: list[float] = []
        with pyBigWig.open(str(path)) as bw:
            chromosomes = bw.chroms()
            for row in rows.itertuples(index=False):
                if row.chrom not in chromosomes:
                    values.append(np.nan)
                    continue
                start = int(row.bin_start)
                end = int(row.bin_end)
                try:
                    stat = bw.stats(row.chrom, start, end, type="mean", exact=True)[0]
                except RuntimeError:
                    stat = None
                values.append(float(stat) if stat is not None else np.nan)
        rows[name] = values
    return rows


def pooled_track(track: pd.DataFrame, value_col: str) -> pd.DataFrame:
    frame = track.copy()
    frame["observed_positions_filled"] = frame["observed_positions"].fillna(0).astype(float)
    frame["weighted"] = frame[value_col].fillna(0).astype(float) * frame["observed_positions_filled"]
    grouped = (
        frame.groupby(KEYS + ["region_name"], as_index=False)
        .agg(
            weighted=("weighted", "sum"),
            observed_positions=("observed_positions_filled", "sum"),
            unique_reads=("unique_reads", "max"),
        )
    )
    grouped[value_col] = np.where(
        grouped["observed_positions"] > 0,
        grouped["weighted"] / grouped["observed_positions"],
        np.nan,
    )
    grouped["sample"] = "pooled"
    return grouped.drop(columns=["weighted"])


def sample_frame(
    sample: str,
    alpha: pd.DataFrame,
    dimelo: pd.DataFrame,
    hyena: pd.DataFrame,
    external: pd.DataFrame,
    alpha_col: str,
    min_reads: int,
    min_positions: int,
) -> pd.DataFrame:
    if sample == "pooled":
        dimelo_sample = pooled_track(dimelo, "mean_signal")
        hyena_sample = pooled_track(hyena, "mean_signal")
    else:
        dimelo_sample = dimelo[dimelo["sample"] == sample].copy()
        hyena_sample = hyena[hyena["sample"] == sample].copy()

    dimelo_sample = dimelo_sample.rename(columns={"mean_signal": "DiMeLo"})
    hyena_sample = hyena_sample.rename(columns={"mean_signal": "HyenaDNA"})
    frame = alpha[KEYS + ["region_name", alpha_col]].rename(columns={alpha_col: "AlphaGenome"})
    frame = frame.merge(external, on=KEYS + ["region_name"], how="inner", validate="one_to_one")
    frame = frame.merge(
        dimelo_sample[KEYS + ["DiMeLo", "unique_reads", "observed_positions"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(
        hyena_sample[KEYS + ["HyenaDNA"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    frame["dimelo_covered"] = (
        frame["DiMeLo"].notna()
        & (frame["unique_reads"].fillna(0) >= min_reads)
        & (frame["observed_positions"].fillna(0) >= min_positions)
    )
    return frame


def evaluate(
    alpha: pd.DataFrame,
    dimelo: pd.DataFrame,
    hyena: pd.DataFrame,
    external: pd.DataFrame,
    external_cols: list[str],
    alpha_col: str,
    min_reads: int,
    min_positions: int,
) -> pd.DataFrame:
    rows = []
    for sample in ("merged_c1", "merged_e5b", "pooled"):
        frame = sample_frame(
            sample, alpha, dimelo, hyena, external, alpha_col, min_reads, min_positions
        )
        for external_col in external_cols:
            external_threshold = float(frame[external_col].dropna().quantile(0.90))
            dimelo_threshold = float(frame.loc[frame["dimelo_covered"], "DiMeLo"].dropna().quantile(0.90))

            comparisons = [
                ("external_bigwig", external_col, "AlphaGenome", "AlphaGenome", False, external_threshold),
                ("external_bigwig", external_col, "HyenaDNA", "HyenaDNA", True, external_threshold),
                ("external_bigwig", external_col, "DiMeLo", "DiMeLo", True, external_threshold),
                ("DiMeLo", "DiMeLo", "AlphaGenome", "AlphaGenome", True, dimelo_threshold),
                ("DiMeLo", "DiMeLo", "HyenaDNA", "HyenaDNA", True, dimelo_threshold),
                ("DiMeLo", "DiMeLo", "external_bigwig", external_col, True, dimelo_threshold),
            ]
            for target_name, target_col, pred_name, pred_col, require_dimelo, threshold in comparisons:
                data = frame[frame["dimelo_covered"]].copy() if require_dimelo else frame.copy()
                data = data[np.isfinite(data[target_col].astype(float)) & np.isfinite(data[pred_col].astype(float))]
                for scope in ["pooled", *sorted(data["chrom"].dropna().unique())]:
                    subset = data if scope == "pooled" else data[data["chrom"] == scope]
                    row = {
                        "scope": scope,
                        "sample": sample,
                        "target": target_name,
                        "target_track": target_col,
                        "prediction": pred_name,
                        "prediction_track": pred_col,
                        "threshold_name": "target_top_10_percent",
                        "threshold": threshold,
                        "number_of_regions": int(subset[["chrom", "region_id"]].drop_duplicates().shape[0]),
                    }
                    row.update(
                        basic_metrics(
                            subset[target_col].to_numpy(float),
                            subset[pred_col].to_numpy(float),
                            threshold,
                        )
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def plot_regions(
    alpha: pd.DataFrame,
    dimelo: pd.DataFrame,
    hyena: pd.DataFrame,
    external: pd.DataFrame,
    external_cols: list[str],
    alpha_col: str,
    out_dir: Path,
    max_regions: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    regions = alpha[["chrom", "region_id", "region_name"]].drop_duplicates().head(max_regions)
    for selected in regions.itertuples(index=False):
        chrom = selected.chrom
        region_id = selected.region_id
        region_name = selected.region_name
        a = alpha[(alpha["chrom"] == chrom) & (alpha["region_id"] == region_id)].copy()
        e = external[(external["chrom"] == chrom) & (external["region_id"] == region_id)].copy()

        fig, axis = plt.subplots(figsize=(13, 5))
        x = (a["bin_start"] + a["bin_end"]) / 2
        axis.plot(x, minmax(a[alpha_col]), color="black", lw=1.5, label="AlphaGenome H3K4me3")
        for external_col in external_cols:
            axis.plot(
                (e["bin_start"] + e["bin_end"]) / 2,
                minmax(e[external_col]),
                color="tab:green",
                lw=1.4,
                alpha=0.85,
                label=f"external bigWig {external_col}",
            )
        for sample, dimelo_color, hyena_color in (
            ("merged_c1", "tab:blue", "tab:purple"),
            ("merged_e5b", "tab:orange", "deeppink"),
        ):
            d = dimelo[(dimelo["chrom"] == chrom) & (dimelo["region_id"] == region_id) & (dimelo["sample"] == sample)]
            h = hyena[(hyena["chrom"] == chrom) & (hyena["region_id"] == region_id) & (hyena["sample"] == sample)]
            axis.plot(
                (d["bin_start"] + d["bin_end"]) / 2,
                minmax(d["mean_signal"]),
                color=dimelo_color,
                alpha=0.8,
                label=f"DiMeLo {sample}",
            )
            axis.plot(
                (h["bin_start"] + h["bin_end"]) / 2,
                minmax(h["mean_signal"]),
                color=hyena_color,
                linestyle="--",
                alpha=0.9,
                label=f"HyenaDNA {sample}",
            )
        axis.set_xlabel(f"{chrom} coordinate (hg38)")
        axis.set_ylabel("min-max normalized signal")
        axis.set_ylim(-0.05, 1.05)
        axis.legend(frameon=False, ncol=2, fontsize=8)
        fig.suptitle(f"{chrom} region {region_id}: external bigWig comparison\n{region_name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{chrom}_region{region_id}_external_bigwig_minmax.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare external GRCh38 bigWig tracks with AlphaGenome, DiMeLo, and HyenaDNA on existing 200 bp bins."
    )
    parser.add_argument("--alpha", type=Path, default=Path("outputs/benchmark_200bp/alphagenome_test_200bp.tsv"))
    parser.add_argument("--dimelo", type=Path, default=Path("outputs/benchmark_200bp/dimelo_test_200bp.tsv"))
    parser.add_argument("--hyena", type=Path, default=Path("outputs/benchmark_200bp/hyenadna_test_200bp.tsv"))
    parser.add_argument("--bigwig", action="append", required=True, help="NAME=/path/file.bigWig, can be repeated.")
    parser.add_argument("--alpha-col", default=ALPHA_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/external_bigwig_benchmark"))
    parser.add_argument("--minimum-unique-reads", type=int, default=2)
    parser.add_argument("--minimum-observed-positions", type=int, default=3)
    parser.add_argument("--max-plotted-regions", type=int, default=12)
    args = parser.parse_args()

    alpha = pd.read_csv(args.alpha, sep="\t")
    dimelo = pd.read_csv(args.dimelo, sep="\t")
    hyena = pd.read_csv(args.hyena, sep="\t")
    if args.alpha_col not in alpha.columns:
        raise SystemExit(f"AlphaGenome column not found: {args.alpha_col}")

    specs = parse_bigwig_specs(args.bigwig)
    external = sample_bigwigs(alpha, specs)
    external_cols = [name for name, _ in specs]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    external.to_csv(args.out_dir / "external_bigwig_200bp.tsv", sep="\t", index=False)
    summary = evaluate(
        alpha,
        dimelo,
        hyena,
        external,
        external_cols,
        args.alpha_col,
        args.minimum_unique_reads,
        args.minimum_observed_positions,
    )
    summary.to_csv(args.out_dir / "external_bigwig_benchmark.summary.tsv", sep="\t", index=False)
    plot_regions(
        alpha,
        dimelo,
        hyena,
        external,
        external_cols,
        args.alpha_col,
        args.out_dir / "normalized_minmax_plots",
        args.max_plotted_regions,
    )
    with open(args.out_dir / "external_bigwig_benchmark.provenance.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "alpha": str(args.alpha),
                "dimelo": str(args.dimelo),
                "hyena": str(args.hyena),
                "bigwigs": {name: str(path) for name, path in specs},
                "bin_size": 200,
                "minimum_unique_reads": args.minimum_unique_reads,
                "minimum_observed_positions": args.minimum_observed_positions,
                "note": "External bigWig signal was used only for population-level evaluation/visualization, not for HyenaDNA training.",
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print(f"Wrote external bigWig benchmark outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
