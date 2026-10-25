"""Exploratory Data Analysis (EDA) script and real Jupyter Notebook executor.

Executes all notebook cells dynamically, captures real stdout/figures,
computes data-derived statistics, and writes a valid executed .ipynb file.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "artifacts" / "figures"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def run_eda():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Starting Exploratory Data Analysis & Real Notebook Execution...")

    parquet_path = DATA_DIR / "hourly_power.parquet"
    stats_path = DATA_DIR / "data_stats.json"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing processed data: {parquet_path}. Run pipeline first.")

    df = pd.read_parquet(parquet_path)
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    # 1. Figure 1: Missingness Over Time
    logger.info("Generating Missingness Analysis Plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    monthly_valid_mins = df["valid_minutes"].resample("ME").sum()
    monthly_expected_mins = df["valid_minutes"].resample("ME").count() * 60
    monthly_missing_pct = (1.0 - monthly_valid_mins / monthly_expected_mins) * 100.0

    ax1.bar(monthly_missing_pct.index, monthly_missing_pct.values, width=20, color="#d62728", alpha=0.8)
    ax1.set_ylabel("Tỷ lệ thiếu phút (%)")
    ax1.set_title("Phân Bố Tỷ Lệ Thiếu Dữ Liệu Theo Tháng (UCI Household Power 2006-2010)")
    ax1.grid(True, linestyle="--", alpha=0.4)

    monthly_valid_hours = df["is_valid_target"].resample("ME").sum()
    monthly_total_hours = df["is_valid_target"].resample("ME").count()
    monthly_valid_ratio = (monthly_valid_hours / monthly_total_hours) * 100.0

    ax2.plot(monthly_valid_ratio.index, monthly_valid_ratio.values, marker="o", color="#1f77b4", linewidth=2.0)
    ax2.set_ylabel("Giờ đủ 60 phút (%)")
    ax2.set_xlabel("Thời gian (Năm)")
    ax2.set_ylim(80, 102)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig1_path = FIGURES_DIR / "eda_missingness.png"
    fig.savefig(fig1_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved: {fig1_path}")

    # 2. Figure 2: Consumption Distribution (Histogram & Percentiles)
    logger.info("Generating Active Power Distribution Plot...")
    valid_power = df["target"].dropna()
    p90 = float(np.percentile(valid_power, 90))
    median_val = float(np.median(valid_power))
    mean_val = float(np.mean(valid_power))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(valid_power, bins=80, density=True, color="#4a7ebb", alpha=0.7, edgecolor="white", label="Phân phối thực tế")
    ax.axvline(median_val, color="#2ca02c", linestyle="-", linewidth=2, label=f"Trung vị (Median): {median_val:.2f} kW")
    ax.axvline(mean_val, color="#ff7f0e", linestyle="--", linewidth=2, label=f"Trung bình (Mean): {mean_val:.2f} kW")
    ax.axvline(p90, color="#d62728", linestyle=":", linewidth=2.2, label=f"Phân vị 90: {p90:.2f} kW")

    ax.set_title("Phân Phối Công Suất Tác Dụng Trung Bình Giờ (Hourly Active Power kW)")
    ax.set_xlabel("Công suất tác dụng (kW)")
    ax.set_ylabel("Mật độ xác suất (Density)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig2_path = FIGURES_DIR / "eda_distribution.png"
    fig.savefig(fig2_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved: {fig2_path}")

    # 3. Figure 3: Diurnal and Weekly Patterns
    logger.info("Generating Diurnal & Weekly Consumption Patterns Plot...")
    hours = df.index.hour
    dayofweek = df.index.day_of_week if hasattr(df.index, "day_of_week") else df.index.dayofweek

    hourly_grp = df.groupby(hours)["raw_active_power"]
    hourly_mean = hourly_grp.mean()
    hourly_p25 = hourly_grp.quantile(0.25)
    hourly_p75 = hourly_grp.quantile(0.75)

    min_hour = int(hourly_mean.idxmin())
    min_hour_val = float(hourly_mean.min())
    max_hour = int(hourly_mean.idxmax())
    max_hour_val = float(hourly_mean.max())

    dow_grp = df.groupby(dayofweek)["raw_active_power"]
    dow_mean = dow_grp.mean()
    weekday_mean = float(df.loc[dayofweek < 5, "raw_active_power"].mean())
    weekend_mean = float(df.loc[dayofweek >= 5, "raw_active_power"].mean())
    dow_labels = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ Nhật"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(hourly_mean.index, hourly_mean.values, color="#1f77b4", linewidth=2.5, marker="o", label="Trung bình giờ")
    ax1.fill_between(hourly_mean.index, hourly_p25.values, hourly_p75.values, color="#1f77b4", alpha=0.25, label="Khoảng Tứ phân vị [P25, P75]")
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xlabel("Giờ trong ngày (0 – 23)")
    ax1.set_ylabel("Công suất tác dụng (kW)")
    ax1.set_title("Biểu Đồ Phụ Tải Ngày Điển Hình (Diurnal Pattern)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.bar(range(7), dow_mean.values, color=["#3274a1" if i < 5 else "#e1812c" for i in range(7)], width=0.6, alpha=0.85)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(dow_labels, rotation=15)
    ax2.set_ylabel("Công suất trung bình (kW)")
    ax2.set_title("Phụ Tải Trung Bình Theo Các Ngày Trong Tuần (Weekly Pattern)")
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig3_path = FIGURES_DIR / "eda_patterns.png"
    fig.savefig(fig3_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved: {fig3_path}")

    # 4. Generate & Actually Execute Jupyter Notebook (.ipynb)
    logger.info("Executing and Generating Jupyter Notebook: notebooks/exploratory_data_analysis.ipynb...")
    execute_and_save_notebook(
        stats=stats,
        mean_val=mean_val,
        median_val=median_val,
        p90=p90,
        min_hour=min_hour,
        min_hour_val=min_hour_val,
        max_hour=max_hour,
        max_hour_val=max_hour_val,
        weekday_mean=weekday_mean,
        weekend_mean=weekend_mean,
    )
    logger.info("EDA completed successfully!")


def execute_and_save_notebook(
    stats: dict,
    mean_val: float,
    median_val: float,
    p90: float,
    min_hour: int,
    min_hour_val: float,
    max_hour: int,
    max_hour_val: float,
    weekday_mean: float,
    weekend_mean: float,
):
    """Construct and execute code cells dynamically, capturing real stdout and figure outputs."""

    # Define cells with code and markdown
    cells_def = [
        {
            "type": "markdown",
            "source": (
                "# Khám Phá Dữ Liệu Tiêu Thụ Điện Hộ Gia Đình (Exploratory Data Analysis)\n"
                "\n"
                "**Bài tập lớn môn học Machine Learning**  \n"
                "**Tập dữ liệu**: UCI Individual Household Electric Power Consumption (Dataset ID: 235)  \n"
                "**Thời gian đo đạc**: 16/12/2006 đến 26/11/2010 (2.075.259 phút đo đạc)\n"
                "\n"
                "> **Lưu ý phương pháp luận**: Phân tích khám phá toàn bộ tập dữ liệu (Full-dataset EDA) ở đây mang tính chất mô tả hậu nghiệm (descriptive posthoc), "
                "phục vụ mục đích hiểu các đặc tính thống kê cơ bản của chuỗi thời gian năng lượng, "
                "hoàn toàn không được sử dụng để tinh chỉnh siêu tham số hay rò rỉ thông tin sang việc đánh giá mô hình trên tập Test."
            ),
        },
        {
            "type": "code",
            "source": (
                "import json\n"
                "from pathlib import Path\n"
                "import pandas as pd\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "\n"
                "# Hoạt động độc lập dù chạy từ thư mục gốc hay thư mục notebooks/\n"
                "data_dir = Path('data/processed') if Path('data/processed').exists() else Path('../data/processed')\n"
                "parquet_path = data_dir / 'hourly_power.parquet'\n"
                "stats_path = data_dir / 'data_stats.json'\n"
                "\n"
                "df = pd.read_parquet(parquet_path)\n"
                "with open(stats_path, 'r', encoding='utf-8') as f:\n"
                "    data_stats = json.load(f)\n"
                "\n"
                "print(f'Tong so khoang gio: {len(df):,}')\n"
                "print(f'So gio du 60 phut hop le: {data_stats[\"complete_target_hours\"]:,} ({data_stats[\"valid_target_ratio\"]:.2%})')\n"
                "print(f'So gio khong hop le: {data_stats[\"incomplete_target_hours\"]:,}')"
            ),
        },
        {
            "type": "markdown",
            "source": (
                "## 1. Thống Kê Mô Tả Công Suất Tác Dụng (Active Power Distribution)\n"
                "\n"
                "Phụ tải điện hộ gia đình có phân phối lệch phải mạnh (Right-skewed), tập trung chủ yếu ở mức phụ tải nền thấp (0,3 - 1,0 kW) "
                "và xuất hiện các đỉnh xung điện ngắn hạn khi bật các thiết bị công suất lớn."
            ),
        },
        {
            "type": "code",
            "source": (
                "valid_power = df['target'].dropna()\n"
                "summary = valid_power.describe(percentiles=[0.25, 0.5, 0.75, 0.90])\n"
                "print('=== THONG KE MO TA CONG SUAT (kW) ===')\n"
                "print(summary.to_string())"
            ),
        },
        {
            "type": "markdown",
            "source": (
                f"- **Giá trị trung vị (Median)**: `{median_val:.4f} kW`\n"
                f"- **Giá trị trung bình (Mean)**: `{mean_val:.4f} kW`\n"
                f"- **Phân vị 90 (P90 High-load)**: `{p90:.4f} kW`"
            ),
        },
        {
            "type": "code",
            "source": (
                "# Ve bieu do phu tai ngay dien hinh (Diurnal Pattern)\n"
                "hours = df.index.hour\n"
                "hourly_mean = df.groupby(hours)['raw_active_power'].mean()\n"
                "\n"
                "fig, ax = plt.subplots(figsize=(10, 4))\n"
                "ax.plot(hourly_mean.index, hourly_mean.values, marker='o', color='#1f77b4', label='Cong suat trung binh (kW)')\n"
                "ax.set_xticks(range(0, 24, 2))\n"
                "ax.set_xlabel('Gio trong ngay (0 - 23)')\n"
                "ax.set_ylabel('Cong suat tac dung (kW)')\n"
                "ax.set_title('Bieu Do Phu Tai Gio Trung Binh Trong Ngay')\n"
                "ax.grid(True, linestyle='--', alpha=0.5)\n"
                "ax.legend()\n"
                "plt.tight_layout()\n"
                "plt.show()"
            ),
        },
        {
            "type": "markdown",
            "source": (
                "## 2. Kết Quả Quan Sát Chu Kỳ Từ Dữ Liệu Thực Tế\n"
                "\n"
                f"- **Khung giờ có phụ tải thấp nhất**: Giờ `{min_hour:02d}:00` đạt mức trung bình `{min_hour_val:.4f} kW`.\n"
                f"- **Khung giờ có phụ tải cao nhất**: Giờ `{max_hour:02d}:00` đạt mức trung bình `{max_hour_val:.4f} kW`.\n"
                f"- **Mức phụ tải trung bình ngày trong tuần (Thứ 2 - Thứ 6)**: `{weekday_mean:.4f} kW`.\n"
                f"- **Mức phụ tải trung bình cuối tuần (Thứ 7 - Chủ Nhật)**: `{weekend_mean:.4f} kW`.\n"
                "\n"
                "Sự chênh lệch này cho thấy tính chu kỳ 24 giờ và tính chất ngày trong tuần là các tín hiệu quan trọng cần đưa vào mô hình học máy."
            ),
        },
    ]

    # Shared execution scope
    exec_scope = {"__name__": "__main__"}
    notebook_cells = []
    exec_count = 1

    for item in cells_def:
        if item["type"] == "markdown":
            notebook_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": item["source"].splitlines(keepends=True),
            })
        elif item["type"] == "code":
            code_text = item["source"]
            stdout_io = io.StringIO()
            captured_outputs = []

            # Execute code cell and capture stdout & plots
            plt.close("all")
            try:
                with contextlib.redirect_stdout(stdout_io):
                    exec(code_text, exec_scope)
            except Exception as e:
                logger.error(f"Error executing notebook cell: {e}")
                raise

            captured_text = stdout_io.getvalue()
            if captured_text:
                captured_outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": captured_text.splitlines(keepends=True),
                })

            # Check if a matplotlib figure was created
            figs = [plt.figure(i) for i in plt.get_fignums()]
            for fig in figs:
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
                buf.seek(0)
                b64_data = base64.b64encode(buf.read()).decode("utf-8")
                plt.close(fig)

                captured_outputs.append({
                    "data": {
                        "image/png": b64_data,
                        "text/plain": ["<Figure size ...>"],
                    },
                    "metadata": {},
                    "output_type": "display_data",
                })

            notebook_cells.append({
                "cell_type": "code",
                "execution_count": exec_count,
                "metadata": {},
                "outputs": captured_outputs,
                "source": code_text.splitlines(keepends=True),
            })
            exec_count += 1

    nb_json = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13.15",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    nb_path = NOTEBOOKS_DIR / "exploratory_data_analysis.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=2)


if __name__ == "__main__":
    run_eda()
