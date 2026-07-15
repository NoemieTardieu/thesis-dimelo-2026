#!/usr/bin/env python3
"""Regression summaries for aggregated 5mC/6mA relationships.

This script complements the paired-read interpretability analysis. The paired
analysis asks whether pairwise read-level 6mA differences can be explained by
pairwise DNA and 5mC differences. Here we ask a coarser question: after
aggregating each unit (window, read-overlap-collapsed read, or region), how
well does mean 5mC explain mean regulatory 6mA?

The model is intentionally simple:

    reg_mean = beta0 + beta1 * m_mean

It reports descriptive metrics on all rows and, when enough groups are present,
leave-one-group-out metrics using chromosome or another requested grouping
column. This is an association analysis, not a causal model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        nargs=2,
        action="append",
        metavar=("UNIT", "TSV"),
        required=True,
        help="Aggregated relationship table, e.g. window path/to/window.tsv.",
    )
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument(
        "--group-col",
        default="chrom",
        help="Column used for leave-one-group-out evaluation. Default: chrom.",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=5,
        help="Minimum rows required for a scope to be evaluated.",
    )
    return parser.parse_args()


def fit_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(x.size), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta


def predict_linear(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return beta[0] + beta[1] * x


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "mae": float(np.mean(np.abs(y - pred))),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "mean_target": float(y.mean()),
        "mean_pred": float(pred.mean()),
    }


def evaluate_subset(
    unit: str,
    scope: str,
    group: str,
    df: pd.DataFrame,
    group_col: str,
    min_n: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    clean = df[["m_mean", "reg_mean", group_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < min_n:
        return rows

    x = clean["m_mean"].to_numpy(float)
    y = clean["reg_mean"].to_numpy(float)
    beta = fit_linear(x, y)
    pred = predict_linear(beta, x)
    descriptive = metric_row(y, pred)
    rows.append(
        {
            "unit": unit,
            "scope": scope,
            "group": group,
            "evaluation": "descriptive_all_rows",
            "n": int(len(clean)),
            "n_groups": int(clean[group_col].nunique()),
            "group_col": group_col,
            "intercept": float(beta[0]),
            "coef_m_mean": float(beta[1]),
            **descriptive,
        }
    )

    groups = list(clean[group_col].dropna().unique())
    if len(groups) >= 2:
        y_all: list[np.ndarray] = []
        pred_all: list[np.ndarray] = []
        betas: list[np.ndarray] = []
        for holdout in groups:
            train = clean[clean[group_col] != holdout]
            test = clean[clean[group_col] == holdout]
            if len(train) < 3 or len(test) == 0:
                continue
            b = fit_linear(train["m_mean"].to_numpy(float), train["reg_mean"].to_numpy(float))
            y_test = test["reg_mean"].to_numpy(float)
            p_test = predict_linear(b, test["m_mean"].to_numpy(float))
            y_all.append(y_test)
            pred_all.append(p_test)
            betas.append(b)
        if y_all:
            y_cv = np.concatenate(y_all)
            pred_cv = np.concatenate(pred_all)
            cv = metric_row(y_cv, pred_cv)
            mean_beta = np.mean(np.vstack(betas), axis=0)
            rows.append(
                {
                    "unit": unit,
                    "scope": scope,
                    "group": group,
                    "evaluation": f"leave_one_{group_col}_out",
                    "n": int(len(y_cv)),
                    "n_groups": int(len(groups)),
                    "group_col": group_col,
                    "intercept": float(mean_beta[0]),
                    "coef_m_mean": float(mean_beta[1]),
                    **cv,
                }
            )
    return rows


def evaluate_table(unit: str, path: Path, group_col: str, min_n: int) -> list[dict[str, object]]:
    df = pd.read_csv(path, sep="\t")
    required = {"m_mean", "reg_mean", group_col}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    rows.extend(evaluate_subset(unit, "all", "all", df, group_col, min_n))
    for col in ["chrom", "sample", "dataset"]:
        if col in df.columns:
            for value, sub in df.groupby(col, dropna=False):
                rows.extend(evaluate_subset(unit, col, str(value), sub, group_col, min_n))
    return rows


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for unit, table in args.table:
        rows.extend(evaluate_table(unit, Path(table), args.group_col, args.min_n))

    out_tsv = out_prefix.with_suffix(".aggregated_regression.tsv")
    out_json = out_prefix.with_suffix(".aggregated_regression.json")
    result = pd.DataFrame(rows)
    result.to_csv(out_tsv, sep="\t", index=False)
    out_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"tsv": str(out_tsv), "json": str(out_json), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
