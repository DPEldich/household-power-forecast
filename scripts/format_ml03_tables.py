"""Format ML-03 metrics tables for reports and verification tests."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def format_uncertainty_markdown_table(unc_csv_path: Path | str) -> str:
    df = pd.read_csv(unc_csv_path)
    lines = [
        "| Split | Phân khúc (Subset) | Số mẫu (Count) | Độ phủ (%) (Coverage) | Dưới P10 (%) | Vượt P90 (%) | Độ rộng TB (kW) | Pinball P10 | Pinball P50 | Pinball P90 | Pinball TB (kW) |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for _, r in df.iterrows():
        sp = str(r["split"])
        sb = str(r["subset"])
        cnt = int(r["count"])
        cov = f"{float(r['coverage_pct']):.2f}%"
        bel = f"{float(r['below_lower_pct']):.2f}%"
        abv = f"{float(r['above_upper_pct']):.2f}%"
        w = f"{float(r['mean_width_kw']):.4f}"
        p10 = f"{float(r['pinball_p10']):.4f}"
        p50 = f"{float(r['pinball_p50']):.4f}"
        p90 = f"{float(r['pinball_p90']):.4f}"
        mp = f"{float(r['mean_pinball']):.4f}"
        lines.append(f"| {sp} | {sb} | {cnt} | {cov} | {bel} | {abv} | {w} | {p10} | {p50} | {p90} | {mp} |")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    csv_p = ROOT / "artifacts/experimental/ml03/metrics/uncertainty_metrics.csv"
    print(format_uncertainty_markdown_table(csv_p))
