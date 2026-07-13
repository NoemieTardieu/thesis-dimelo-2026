#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path


METRICS = ("pearson", "spearman", "auroc", "auprc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create readable AlphaGenome-vs-HyenaDNA metric summaries."
    )
    parser.add_argument(
        "--metrics-tsv",
        default="outputs/promoter_followup/alphagenome_vs_hyenadna_primary_metrics.tsv",
    )
    parser.add_argument(
        "--out-prefix",
        default="outputs/promoter_followup/alphagenome_vs_hyenadna_readable_summary",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def delta_class(delta: float) -> str:
    if delta >= 0.05:
        return "hyena-strong"
    if delta >= 0.01:
        return "hyena"
    if delta <= -0.05:
        return "alpha-strong"
    if delta <= -0.01:
        return "alpha"
    return "tie"


def winner(delta: float) -> str:
    if delta > 0.01:
        return "HyenaDNA"
    if delta < -0.01:
        return "AlphaGenome"
    return "similar"


def model_pairs(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row["experimental_sample"], row["scope"])
        grouped[key][row["model"]] = row

    pairs = []
    order = {"pooled": 0, "chr11": 1, "chr16": 2, "chr17": 3, "chr19": 4}
    for (sample, scope), models in grouped.items():
        if "AlphaGenome" not in models or "HyenaDNA" not in models:
            continue
        alpha = models["AlphaGenome"]
        hyena = models["HyenaDNA"]
        pair: dict[str, object] = {
            "experimental_sample": sample,
            "scope": scope,
            "number_of_bins": int(float(hyena["number_of_bins"])),
            "positive_fraction": f(hyena, "positive_fraction"),
        }
        for metric in METRICS:
            alpha_value = f(alpha, metric)
            hyena_value = f(hyena, metric)
            pair[f"alpha_{metric}"] = alpha_value
            pair[f"hyena_{metric}"] = hyena_value
            pair[f"delta_{metric}"] = hyena_value - alpha_value
            pair[f"winner_{metric}"] = winner(hyena_value - alpha_value)
        pairs.append(pair)
    return sorted(
        pairs,
        key=lambda row: (str(row["experimental_sample"]), order.get(str(row["scope"]), 99)),
    )


def write_delta_tsv(pairs: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "experimental_sample",
        "scope",
        "number_of_bins",
        "positive_fraction",
    ]
    for metric in METRICS:
        fieldnames.extend(
            [
                f"alpha_{metric}",
                f"hyena_{metric}",
                f"delta_hyena_minus_alpha_{metric}",
                f"winner_{metric}",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            out = {
                "experimental_sample": pair["experimental_sample"],
                "scope": pair["scope"],
                "number_of_bins": pair["number_of_bins"],
                "positive_fraction": fmt(float(pair["positive_fraction"])),
            }
            for metric in METRICS:
                out[f"alpha_{metric}"] = fmt(float(pair[f"alpha_{metric}"]))
                out[f"hyena_{metric}"] = fmt(float(pair[f"hyena_{metric}"]))
                out[f"delta_hyena_minus_alpha_{metric}"] = fmt(float(pair[f"delta_{metric}"]))
                out[f"winner_{metric}"] = pair[f"winner_{metric}"]
            writer.writerow(out)


def summarize_pooled(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    return [pair for pair in pairs if pair["scope"] == "pooled"]


def write_markdown(pairs: list[dict[str, object]], path: Path) -> None:
    pooled = summarize_pooled(pairs)
    lines = [
        "# AlphaGenome vs HyenaDNA: Readable Interpretation",
        "",
        "This summary compares both models on the same averaged/binned DiMeLo 6mA target.",
        "Positive deltas mean HyenaDNA is higher; negative deltas mean AlphaGenome is higher.",
        "",
        "## Main Takeaway",
        "",
        "- HyenaDNA is usually better for continuous agreement with the averaged signal (`pearson`, `spearman`).",
        "- AlphaGenome is often competitive or better for peak-like binary retrieval (`auprc`), especially in pooled/e5b summaries.",
        "- AUROC is very similar between models; neither model dominates every metric.",
        "- This means the two models are useful in slightly different ways: HyenaDNA tracks the experimental signal shape better, while AlphaGenome can be strong for identifying high-signal bins.",
        "",
        "## Pooled Summary",
        "",
        "| sample | metric | AlphaGenome | HyenaDNA | Hyena - Alpha | better |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for pair in pooled:
        sample = str(pair["experimental_sample"])
        for metric in METRICS:
            delta = float(pair[f"delta_{metric}"])
            lines.append(
                "| "
                + " | ".join(
                    [
                        sample,
                        metric,
                        fmt(float(pair[f"alpha_{metric}"])),
                        fmt(float(pair[f"hyena_{metric}"])),
                        fmt(delta),
                        str(pair[f"winner_{metric}"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Suggested Text To Send",
            "",
            "The averaged-signal comparison shows that HyenaDNA and AlphaGenome capture related but not identical aspects of the DiMeLo-derived regulatory signal. HyenaDNA generally achieves higher Pearson and Spearman correlations, suggesting better agreement with the continuous averaged 6mA signal. AlphaGenome is competitive for AUROC and often stronger for AUPRC, suggesting good enrichment for the highest-signal bins. Overall, this supports that the HyenaDNA model has learned sequence-associated regulatory signal, while AlphaGenome provides a useful SoTA reference and complementary benchmark.",
            "",
            "## Files",
            "",
            "- `alphagenome_vs_hyenadna_primary_metrics.tsv`: full raw metrics with confidence intervals.",
            "- `alphagenome_vs_hyenadna_readable_summary.deltas.tsv`: compact model-vs-model deltas.",
            "- `alphagenome_vs_hyenadna_readable_summary.colored.html`: color-coded table for easy visual inspection.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(pairs: list[dict[str, object]], path: Path) -> None:
    css = """
body { font-family: Arial, sans-serif; margin: 28px; color: #202124; }
table { border-collapse: collapse; font-size: 13px; }
th, td { border: 1px solid #d0d7de; padding: 6px 8px; text-align: right; }
th { background: #f6f8fa; position: sticky; top: 0; }
td.left, th.left { text-align: left; }
.hyena-strong { background: #b7e4c7; }
.hyena { background: #d8f3dc; }
.alpha-strong { background: #f4b6b6; }
.alpha { background: #ffd6d6; }
.tie { background: #f1f3f4; }
.note { max-width: 980px; line-height: 1.45; }
"""
    metric_headers = []
    for metric in METRICS:
        metric_headers.extend(
            [
                f"Alpha {metric}",
                f"Hyena {metric}",
                f"Delta {metric}",
                f"Better {metric}",
            ]
        )
    parts = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>AlphaGenome vs HyenaDNA readable metrics</title>",
        f"<style>{css}</style>",
        "</head>",
        "<body>",
        "<h1>AlphaGenome vs HyenaDNA: averaged-signal metrics</h1>",
        "<p class='note'>Green cells mean HyenaDNA is higher; red cells mean AlphaGenome is higher. Darker color means the difference is at least 0.05. The most compact interpretation: HyenaDNA usually wins continuous correlation; AlphaGenome is often stronger for AUPRC/peak-like retrieval.</p>",
        "<table>",
        "<tr>",
        "<th class='left'>sample</th><th class='left'>scope</th><th>bins</th><th>pos frac</th>",
    ]
    for header in metric_headers:
        parts.append(f"<th>{html.escape(header)}</th>")
    parts.append("</tr>")
    for pair in pairs:
        parts.append("<tr>")
        parts.append(f"<td class='left'>{html.escape(str(pair['experimental_sample']))}</td>")
        parts.append(f"<td class='left'>{html.escape(str(pair['scope']))}</td>")
        parts.append(f"<td>{pair['number_of_bins']}</td>")
        parts.append(f"<td>{fmt(float(pair['positive_fraction']))}</td>")
        for metric in METRICS:
            delta = float(pair[f"delta_{metric}"])
            klass = delta_class(delta)
            parts.append(f"<td>{fmt(float(pair[f'alpha_{metric}']))}</td>")
            parts.append(f"<td>{fmt(float(pair[f'hyena_{metric}']))}</td>")
            parts.append(f"<td class='{klass}'>{fmt(delta)}</td>")
            parts.append(f"<td class='{klass}'>{html.escape(str(pair[f'winner_{metric}']))}</td>")
        parts.append("</tr>")
    parts.extend(["</table>", "</body>", "</html>"])
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    metrics_path = Path(args.metrics_tsv)
    if not metrics_path.is_absolute():
        metrics_path = script_dir / metrics_path
    out_prefix = Path(args.out_prefix)
    if not out_prefix.is_absolute():
        out_prefix = script_dir / out_prefix
    rows = read_rows(metrics_path)
    pairs = model_pairs(rows)
    write_delta_tsv(pairs, out_prefix.with_suffix(".deltas.tsv"))
    write_markdown(pairs, out_prefix.with_suffix(".interpretation.md"))
    write_html(pairs, out_prefix.with_suffix(".colored.html"))
    print(
        "\n".join(
            [
                f"Wrote {out_prefix.with_suffix('.deltas.tsv')}",
                f"Wrote {out_prefix.with_suffix('.interpretation.md')}",
                f"Wrote {out_prefix.with_suffix('.colored.html')}",
            ]
        )
    )


if __name__ == "__main__":
    main()
