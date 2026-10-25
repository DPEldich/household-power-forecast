"""Report and README generator that binds all tables directly to CSV and metadata artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT_DIR / "artifacts" / "metrics"
METADATA_DIR = ROOT_DIR / "artifacts" / "metadata"


def generate_reports():
    # 1. Load data
    with open(METADATA_DIR / "split_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    test_df = pd.read_csv(METRICS_DIR / "test_metrics.csv")
    val_df = pd.read_csv(METRICS_DIR / "validation_metrics.csv")
    cv_df = pd.read_csv(METRICS_DIR / "cv_metrics.csv")
    val_ab_df = pd.read_csv(METRICS_DIR / "ablation_metrics.csv")
    test_ab_df = pd.read_csv(METRICS_DIR / "test_ablation_metrics.csv")
    hl_df = pd.read_csv(METRICS_DIR / "high_load_metrics.csv")

    # Format helpers
    def f4(v):
        return f"{float(v):.4f}"

    def f2p(v):
        return f"{float(v):+.2f}%"

    def f2(v):
        return f"{float(v):.2f}"

    # Build CV Table markdown
    # Group by model_name and pick selected config
    best_cfgs = meta["best_hyperparameters"]
    cv_rows_md = []
    for model_name, cfg in best_cfgs.items():
        row = cv_df[(cv_df["model_name"] == model_name) & (cv_df["config"] == str(cfg))].iloc[0]
        cfg_str = str(cfg).replace("{", "").replace("}", "").replace("'", "")
        cv_rows_md.append(
            f"| **{model_name}** | `{cfg_str}` | {f4(row['mean_cv_mae'])} | {f4(row['std_cv_mae'])} | {f4(row['mean_cv_rmse'])} | {f4(row['mean_cv_r2'])} |"
        )
    cv_table_md = "\n".join(cv_rows_md)

    # Build Validation Table markdown
    val_rows_md = []
    for _, r in val_df.iterrows():
        val_rows_md.append(
            f"| {r['model_name']} | {f4(r['mae'])} | {f4(r['rmse'])} | {f4(r['r2'])} | {f2p(r['mae_improvement_vs_persistence_pct'])} | {f2p(r['mae_improvement_vs_seasonal_naive_pct'])} | {f2(r['fit_time_seconds'])} |"
        )
    val_table_md = "\n".join(val_rows_md)

    # Build Test Table markdown
    test_rows_md = []
    readme_test_rows_md = []
    for i, r in test_df.iterrows():
        bold = "**" if r["model_name"] == meta["validation_selected_winner"] else ""
        test_rows_md.append(
            f"| {i+1} | {bold}{r['model_name']}{bold} | {bold}{f4(r['mae'])}{bold} | {bold}{f4(r['rmse'])}{bold} | {bold}{f4(r['r2'])}{bold} | {bold}{f2p(r['mae_improvement_vs_persistence_pct'])}{bold} | {bold}{f2p(r['mae_improvement_vs_seasonal_naive_pct'])}{bold} | {bold}{f2(r['fit_time_seconds'])}{bold} | {r['sample_count']:,} |"
        )
        readme_test_rows_md.append(
            f"| {i+1} | {bold}{r['model_name']}{bold} | {bold}{f4(r['mae'])}{bold} | {bold}{f4(r['rmse'])}{bold} | {bold}{f4(r['r2'])}{bold} | {bold}{f2p(r['mae_improvement_vs_persistence_pct'])}{bold} | {bold}{f2p(r['mae_improvement_vs_seasonal_naive_pct'])}{bold} | {bold}{f2(r['fit_time_seconds'])}{bold} |"
        )
    test_table_md = "\n".join(test_rows_md)
    readme_test_table_md = "\n".join(readme_test_rows_md)

    # Build Ablation Table markdown
    ab_map = {
        "calendar_only": "1. Calendar Only (Chi Lịch)",
        "lag_rolling_only": "2. Lag & Rolling Only (Chi Chuoi)",
        "full_minus_rolling": "3. Full minus Rolling (Lich + Lags)",
        "full": "4. Full Features (Toan bo 20 dac trung)",
    }
    ab_rows_md = []
    for _, r_val in val_ab_df.iterrows():
        cfg_name = r_val["config_name"]
        r_test = test_ab_df[test_ab_df["config_name"] == cfg_name].iloc[0]
        desc = ab_map.get(cfg_name, cfg_name)
        bold = "**" if cfg_name == "full" else ""
        ab_rows_md.append(
            f"| {bold}{desc}{bold} | {r_val['num_features']} | {bold}{f4(r_val['mae'])}{bold} | {f4(r_val['rmse'])} | {f4(r_val['r2'])} | {bold}{f4(r_test['mae'])}{bold} | {f4(r_test['rmse'])} | {f4(r_test['r2'])} |"
        )
    ab_table_md = "\n".join(ab_rows_md)

    # Build High Load Table markdown
    hl_rows_md = []
    for _, r in hl_df.iterrows():
        hl_rows_md.append(
            f"| **{r['model_name']}** | {f4(r['high_load_mae'])} | {f4(r['high_load_rmse'])} | {f4(r['normal_load_mae'])} | {f4(r['normal_load_rmse'])} | {f4(r['overall_mae'])} |"
        )
    hl_table_md = "\n".join(hl_rows_md)

    # Metadata values
    dstats = meta["dataset_stats"]
    splits = meta["splits"]
    vmask = meta["val_common_mask_stats"]
    tmask = meta["test_common_mask_stats"]

    # Generate REPORT.md
    report_content = f"""# BÁO CÁO KẾT THÚC HỌC PHẦN MACHINE LEARNING
## ĐỀ TÀI: DỰ BÁO CÔNG SUẤT TIÊU THỤ ĐIỆN HỘ GIA ĐÌNH THEO GIỜ TIẾP THEO (NEXT-HOUR HOUSEHOLD ACTIVE POWER FORECASTING)

---

## Tóm Tắt Đồ Án (Executive Summary)

Đề tài giải quyết bài toán dự báo chuỗi thời gian phụ tải điện gia đình: **ước lượng công suất tác dụng trung bình trong 1 giờ tiếp theo ($y_s$, đơn vị kW)** của một hộ gia đình cho khoảng thời gian $[s, s+1\\text{{h}})$, sử dụng thông tin lịch sử diễn ra trước thời điểm $s$ ($< s$).

Nghiên cứu sử dụng tập dữ liệu đo đạc thực tế **UCI Individual Household Electric Power Consumption** (Dataset 235), gồm **{dstats['total_minute_rows']:,} phút đo đạc** trong giai đoạn từ tháng 12/2006 đến tháng 11/2010 ({dstats['first_minute_timestamp'][:10]} đến {dstats['last_minute_timestamp'][:10]}).

Quy trình nghiên cứu áp dụng các nguyên tắc về tính toàn vẹn nhân quả (Strict Causality) và kiểm định học máy chuẩn mực:
1. **Phân chia thời gian**: {splits['warmup_hours']} giờ khởi động (Warmup), 70% thời gian đầu làm tập Train (chứa 3-fold Expanding Temporal Cross-Validation), 15% thời gian tiếp theo làm tập Validation (để chọn cấu hình tối ưu), và 15% thời gian cuối cùng làm tập Test.
2. **Khảo sát 7 mô hình**: 2 mô hình chuẩn cơ sở (Persistence, Seasonal Naive), 2 mô hình hồi quy tuyến tính (Linear Regression, Ridge Regression) và 3 mô hình học máy dạng cây (Random Forest, Gradient Boosting, XGBoost).
3. **Mặt nạ đánh giá chung (Common Mask)**: Toàn bộ 7 mô hình được đánh giá trên cùng một tập chỉ mục thời gian đồng nhất ({tmask['common_valid_hours']:,} giờ kiểm định độc lập).
4. **Kết quả nổi bật**: Mô hình **{meta['validation_selected_winner']}** đạt sai số nhỏ nhất trên tập Validation (MAE = {f4(val_df[val_df['model_name'] == meta['validation_selected_winner']]['mae'].iloc[0])} kW) và được lựa chọn làm mô hình đại diện. Trên tập Test độc lập, {meta['validation_selected_winner']} đạt **MAE = {f4(test_df[test_df['model_name'] == meta['validation_selected_winner']]['mae'].iloc[0])} kW**, **RMSE = {f4(test_df[test_df['model_name'] == meta['validation_selected_winner']]['rmse'].iloc[0])} kW**, $R^2 = {f4(test_df[test_df['model_name'] == meta['validation_selected_winner']]['r2'].iloc[0])}$.

---

## 1. Phương Pháp Luận & Quy Ước Bài Toán

### 1.1. Quy Ước Thời Gian & Tính Nhân Quả (Causal Integrity)
- Mỗi hàng dữ liệu được đánh chỉ mục bởi thời điểm bắt đầu khoảng mục tiêu $s$.
- Giá trị mục tiêu $y_s$ là công suất tác dụng trung bình trong khoảng 1 giờ $[s, s+1\\text{{h}})$, được tổng hợp theo quy ước nhãn trái (`closed='left'`, `label='left'`).
- Thời điểm phát dự báo là $s$. Toàn bộ đặc trưng đầu vào $X_s$ chỉ được tính toán từ các quan sát diễn ra trước $s$ (tức các giờ $\\le s-1$).
- Chuỗi thời gian được thiết lập trên chỉ mục liên tục hàng giờ (`pd.date_range(..., freq='1h')`) trước khi tạo bất kỳ biến trễ hay biến trượt nào, đảm bảo khoảng cách vật lý thực tế không bị co giãn khi gặp các khoảng thời gian gián đoạn đo đạc.

### 1.2. Phân Biệt Khái Niệm Công Suất (kW) và Điện Năng (kWh)
- **Công suất tác dụng ($P$, kW)**: Đại lượng đo tốc độ chuyển hóa điện năng tức thời. Trong bài toán này, giá trị đo đạc là trung bình số học của các phút trong giờ.
- **Điện năng tiêu thụ ($E$, kWh)**: Năng lượng tích lũy tiêu hao trong khoảng thời gian $t$:
  $$E = \\int_{{0}}^{{t}} P(\\tau) d\\tau \\approx P_{{\\text{{tb}}}} \\times t$$
  Vì khoảng thời gian dự báo ở đây cố định bằng đúng 1 giờ ($t = 1\\text{{ h}}$), giá trị công suất trung bình $1.5\\text{{ kW}}$ tương ứng với $1.5\\text{{ kWh}}$ điện năng tiêu thụ trong khung giờ đó.

---

## 2. Kiểm Soát Chất Lượng Dữ Liệu & Tiền Xử Lý

### 2.1. Thống Kê Dữ Liệu Gốc (Data Quality Statistics)
- Tổng số bản ghi phút thô: **{dstats['total_minute_rows']:,} phút**.
- Số phút thiếu đo đạc công suất (`?` hoặc rỗng): **{dstats['missing_minute_readings']:,} phút ({dstats['missing_minute_readings']/dstats['total_minute_rows']:.2%})**.
- Số bản ghi trùng lặp timestamp: **{dstats['duplicate_minute_timestamps']}**.
- Mốc thời gian bắt đầu: `{dstats['first_minute_timestamp']}`.
- Mốc thời gian kết thúc: `{dstats['last_minute_timestamp']}`.

### 2.2. Chuẩn Hóa Chuỗi Giờ & Tiêu Chuẩn Nhãn Thực Tế
Sau khi tổng hợp sang tần suất 1 giờ liên tục:
- Tổng số khoảng 1 giờ liên tục: **{dstats['total_hourly_intervals']:,} giờ**.
- **Tiêu chuẩn nhãn thực tế hợp lệ**: Một giờ $s$ chỉ được coi là có nhãn ground truth hợp lệ ($y_s$) khi có **đủ đúng 60 phút đo đạc không bị thiếu**.
  - Số giờ hợp lệ đạt chuẩn: **{dstats['complete_target_hours']:,} giờ ({dstats['valid_target_ratio']:.2%})**.
  - Số giờ không hợp lệ (có $\\le 59$ phút đo đạc): **{dstats['incomplete_target_hours']:,} giờ**.
- **Quy tắc bảo tồn dữ liệu**: Toàn bộ {dstats['incomplete_target_hours']} giờ không đủ 60 phút được gán $y_s = \\text{{NaN}}$ và bị loại khỏi việc tính điểm đánh giá; không tổng hợp nhãn giả mạo.
- **Xử lý chuỗi đầu vào lịch sử (Causal Imputation)**: Đối với chuỗi đầu vào dùng để tính đặc trưng trễ, các khoảng thiếu được điền tiến có chặn (bounded forward-fill tối đa 3 giờ liên tiếp `ffill(limit=3)`). Đối với các giá trị thiếu còn lại sau 3 giờ, sử dụng `SimpleImputer(strategy='median')` được huấn luyện **độc quyền trên tập Train của từng fold**, không làm rò rỉ dữ liệu sang tập Validation hay Test. Do đó, các biến trễ baseline phản ánh chuỗi lịch sử được điền tiến có giới hạn này.

---

## 3. Kỹ Thuật Tạo Đặc Trưng Nhân Quả (Feature Engineering)

Toàn bộ 20 đặc trưng được thiết kế tuân thủ nguyên tắc không rò rỉ dữ liệu:

1. **Đặc trưng độ trễ ngắn hạn và dài hạn (Lag Features - 6 biến)**:
   - Trễ 1, 2, 3 giờ: $x_{{s-1}}, x_{{s-2}}, x_{{s-3}}$.
   - Trễ 24, 48 giờ: $x_{{s-24}}, x_{{s-48}}$.
   - Trễ 168 giờ: $x_{{s-168}}$.
2. **Đặc trưng thống kê trượt (Rolling Statistics - 6 biến)**:
   - Được tính toán trên chuỗi đã dịch chuyển 1 giờ (`shift(1)`), đảm bảo giá trị của giờ hiện tại $s$ không tham gia vào cửa sổ trượt:
     - Trung bình trượt: `rolling_mean_3`, `rolling_mean_24`, `rolling_mean_168`.
     - Độ lệch chuẩn trượt: `rolling_std_3`, `rolling_std_24`, `rolling_std_168`.
3. **Đặc trưng lịch và tuần hoàn (Calendar & Cyclic - 8 biến)**:
   - Biến số: `hour` ($0..23$), `dayofweek` ($0..6$), `month` ($1..12$), `is_weekend` ($0/1$).
   - Biến tuần hoàn hình sin/cosin để giữ tính liên tục tại điểm chuyển giao 23:59 sang 00:00:
     $$\\sin\\left(\\frac{{2\\pi \\cdot \\text{{hour}}}}{{24}}\\right), \\quad \\cos\\left(\\frac{{2\\pi \\cdot \\text{{hour}}}}{{24}}\\right), \\quad \\sin\\left(\\frac{{2\\pi \\cdot \\text{{dayofweek}}}}{{7}}\\right), \\quad \\cos\\left(\\frac{{2\\pi \\cdot \\text{{dayofweek}}}}{{7}}\\right)$$

*Giai đoạn khởi động (Warmup)*: {splits['warmup_hours']} giờ đầu tiên của tập dữ liệu được loại trừ hoàn toàn khỏi mọi mô hình để đảm bảo các biến trễ $s-168$ luôn có đủ dữ liệu lịch sử xác thực.

---

## 4. Thiết Kế Phân Chia Dữ Liệu & Kiểm Định Mở Rộng

### 4.1. Phân Chia Thời Gian (Chronological Split Boundaries)
Các ranh giới được ấn định theo mốc thời gian lịch trước khi lọc các nhãn không hợp lệ:

| Phân Vùng | Tỷ Lệ | Mốc Bắt Đầu | Mốc Kết Thúc | Tổng Giờ Lịch | Giờ Đủ 60 Phút Hợp Lệ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Warmup** | - | 2006-12-16 17:00:00 | 2006-12-23 16:00:00 | {splits['warmup_hours']} | - |
| **Train Set** | 70% | {splits['train_boundary']['start_timestamp']} | {splits['train_boundary']['end_timestamp']} | {splits['train_boundary']['total_hours']:,} | {splits['train_boundary']['valid_target_hours']:,} |
| **Validation Set** | 15% | {splits['val_boundary']['start_timestamp']} | {splits['val_boundary']['end_timestamp']} | {splits['val_boundary']['total_hours']:,} | {splits['val_boundary']['valid_target_hours']:,} |
| **First 85% (Refit)** | 85% | {splits['first85_boundary']['start_timestamp']} | {splits['first85_boundary']['end_timestamp']} | {splits['first85_boundary']['total_hours']:,} | {splits['first85_boundary']['valid_target_hours']:,} |
| **Test Set** | 15% | {splits['test_boundary']['start_timestamp']} | {splits['test_boundary']['end_timestamp']} | {splits['test_boundary']['total_hours']:,} | {splits['test_boundary']['valid_target_hours']:,} |

### 4.2. Ranh Giới 3 Fold Expanding Cross-Validation Trong Tập Train (70%)
- **Fold 1**: Train [`{meta['cv_folds'][0]['train_start']}` $\\rightarrow$ `{meta['cv_folds'][0]['train_end']}`] ({meta['cv_folds'][0]['train_valid_hours']:,} mẫu) | Val [`{meta['cv_folds'][0]['val_start']}` $\\rightarrow$ `{meta['cv_folds'][0]['val_end']}`] ({meta['cv_folds'][0]['val_valid_hours']:,} mẫu).
- **Fold 2**: Train [`{meta['cv_folds'][1]['train_start']}` $\\rightarrow$ `{meta['cv_folds'][1]['train_end']}`] ({meta['cv_folds'][1]['train_valid_hours']:,} mẫu) | Val [`{meta['cv_folds'][1]['val_start']}` $\\rightarrow$ `{meta['cv_folds'][1]['val_end']}`] ({meta['cv_folds'][1]['val_valid_hours']:,} mẫu).
- **Fold 3**: Train [`{meta['cv_folds'][2]['train_start']}` $\\rightarrow$ `{meta['cv_folds'][2]['train_end']}`] ({meta['cv_folds'][2]['train_valid_hours']:,} mẫu) | Val [`{meta['cv_folds'][2]['val_start']}` $\\rightarrow$ `{meta['cv_folds'][2]['val_end']}`] ({meta['cv_folds'][2]['val_valid_hours']:,} mẫu).

### 4.3. Mặt Nạ Đánh Giá Chung (Common Evaluation Mask)
Để so sánh công bằng giữa các mô hình học máy và mô hình chuẩn cơ sở:
- Một mốc thời gian $s$ nằm trong tập đánh giá chung khi và chỉ khi: (1) Nhãn $y_s$ có đủ 60 phút hợp lệ; (2) Tồn tại đầy đủ $x_{{s-1}}$ (cho Persistence) và $x_{{s-24}}$ (cho Seasonal Naive).
- **Trên tập Validation**:
  - Số giờ lịch: {vmask['calendar_hours']:,} giờ.
  - Số giờ nhãn không hợp lệ: {vmask['invalid_target_hours']:,} giờ.
  - Số giờ nhãn hợp lệ: {vmask['valid_target_hours']:,} giờ.
  - Số giờ bị loại do thiếu baseline: {vmask['baseline_excluded_hours']:,} giờ.
  - **Mặt nạ chung (Common Mask)**: **{vmask['common_valid_hours']:,} giờ** (Đạt tỷ lệ bao phủ {vmask['calendar_coverage_pct']:.2f}% giờ lịch và {vmask['valid_target_coverage_pct']:.2f}% giờ nhãn hợp lệ).
- **Trên tập Test**:
  - Số giờ lịch: {tmask['calendar_hours']:,} giờ.
  - Số giờ nhãn không hợp lệ: {tmask['invalid_target_hours']:,} giờ.
  - Số giờ nhãn hợp lệ: {tmask['valid_target_hours']:,} giờ.
  - Số giờ bị loại do thiếu baseline: {tmask['baseline_excluded_hours']:,} giờ.
  - **Mặt nạ chung (Common Mask)**: **{tmask['common_valid_hours']:,} giờ** (Đạt tỷ lệ bao phủ {tmask['calendar_coverage_pct']:.2f}% giờ lịch và {tmask['valid_target_coverage_pct']:.2f}% giờ nhãn hợp lệ).

---

## 5. Kết Quả Huấn Luyện & Thẩm Định Mô Hình

### 5.1. Kết Quả Kiểm Định Chéo Mở Rộng (Train Expanding CV)
Bảng kết quả trung bình và độ lệch chuẩn qua 3 fold mở rộng trên tập Train (trích xuất trực tiếp từ `cv_metrics.csv`):

| Mô Hình | Cấu Hình Tối Ưu Lựa Chọn | Mean CV MAE (kW) | Std CV MAE | Mean CV RMSE (kW) | Mean CV $R^2$ |
|:---|:---|:---:|:---:|:---:|:---:|
{cv_table_md}

### 5.2. Đánh Giá Trên Tập Validation & Lựa Chọn Mô Hình Chiến Thắng
Mô hình sau khi được cố định siêu tham số tốt nhất từ CV được huấn luyện trên toàn bộ tập Train (70%) và đánh giá trên tập Validation (15%):

| Mô Hình | Validation MAE (kW) | Validation RMSE (kW) | Validation $R^2$ | % Cải Thiện vs Persistence | % Cải Thiện vs Seasonal Naive | Thời Gian Fit (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
{val_table_md}

> **Quyết định thẩm định**: **{meta['validation_selected_winner']}** đạt MAE nhỏ nhất ({f4(val_df[val_df['model_name'] == meta['validation_selected_winner']]['mae'].iloc[0])} kW) trên tập Validation, được chính thức lựa chọn làm **Validation Winner** trước khi bước vào đánh giá trên tập Test.

---

## 6. Kết Quả Đánh Giá Trên Tập Test (Held-Out Test Results)

Mô hình được tái huấn luyện trên 85% dữ liệu đầu (Train + Val, gồm {splits['first85_boundary']['valid_target_hours']:,} giờ hợp lệ) với cấu hình siêu tham số cố định và đánh giá trên tập Test ({tmask['common_valid_hours']:,} giờ kiểm định). Lưu ý rằng trong quá trình phát triển dự án, việc thực thi lại quy trình đã được thực hiện phục vụ mục đích kiểm định kỹ thuật mã nguồn mà không thay đổi bất kỳ siêu tham số nào theo tập test.

| STT | Mô Hình | MAE (kW) | RMSE (kW) | $R^2$ Score | Cải Thiện vs Persistence | Cải Thiện vs Seasonal Naive | Thời Gian Fit (s) | Số Mẫu Test |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
{test_table_md}

### Nhận Xét Kết Quả Chính:
1. **Mô hình học máy dạng cây đạt kết quả vượt trội**: {meta['validation_selected_winner']} đạt MAE {f4(test_df[test_df['model_name'] == meta['validation_selected_winner']]['mae'].iloc[0])} kW, cải thiện {f2p(test_df[test_df['model_name'] == meta['validation_selected_winner']]['mae_improvement_vs_persistence_pct'].iloc[0])} so với mô hình Persistence và {f2p(test_df[test_df['model_name'] == meta['validation_selected_winner']]['mae_improvement_vs_seasonal_naive_pct'].iloc[0])} so với Seasonal Naive.
2. **Persistence tốt hơn Seasonal Naive**: Mô hình Persistence (lấy giờ liền trước $s-1$, MAE {f4(test_df[test_df['model_name'] == 'Persistence']['mae'].iloc[0])} kW) tốt hơn Seasonal Naive (lấy cùng giờ ngày hôm trước $s-24$, MAE {f4(test_df[test_df['model_name'] == 'Seasonal Naive']['mae'].iloc[0])} kW) {f2p(test_df[test_df['model_name'] == 'Persistence']['mae_improvement_vs_seasonal_naive_pct'].iloc[0])}. Điều này cho thấy tính tương quan tự hồi quy cực ngắn hạn (1 giờ trước) mạnh hơn so với tính chu kỳ 24 giờ của một hộ gia đình đơn lẻ.

---

## 7. Nghiên Cứu Cắt Bỏ Thành Phần (Ablation Study)

Thử nghiệm Ablation được tiến hành trên mô hình **{meta['validation_selected_winner']}** trên tập Validation và tập Test với thứ tự các cột đặc trưng đồng nhất hoàn toàn:

| Nhóm Cấu Hình | Số Biến | Val MAE (kW) | Val RMSE (kW) | Val $R^2$ | Test MAE (kW) | Test RMSE (kW) | Test $R^2$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
{ab_table_md}

### Đóng Góp Của Từng Nhóm Đặc Trưng:
- **Vai trò của nhóm trễ và chuỗi thời gian**: Khi chỉ sử dụng các biến lịch (cấu hình `Calendar Only`), sai số MAE trên Validation tăng lên {f4(val_ab_df[val_ab_df['config_name'] == 'calendar_only']['mae'].iloc[0])} kW (và trên Test là {f4(test_ab_df[test_ab_df['config_name'] == 'calendar_only']['mae'].iloc[0])} kW), cho thấy nhóm thông tin trễ và trượt chuỗi thời gian đóng vai trò quyết định.
- **Sự kết hợp đầy đủ**: Trên tập Test, cấu hình đầy đủ `Full Features` đạt MAE {f4(test_ab_df[test_ab_df['config_name'] == 'full']['mae'].iloc[0])} kW, cho thấy việc kết hợp cả đặc trưng trễ, đặc trưng trượt và đặc trưng lịch tuần hoàn mang lại độ chính xác tổng thể tốt nhất.

---

## 8. Phân Tích Sai Số Phụ Tải Cao (High-Load Error Analysis)

Ngưỡng phụ tải cao được xác định tại **phân vị 90 (P90) của tập Train**: **{f4(meta['high_load_threshold_train_p90_kW'])} kW**.

Đánh giá độ chính xác của các mô hình trên tập Test được phân tách thành hai miền phụ tải:

| Mô Hình | Giờ Cao Điểm ($\\\\ge {f4(meta['high_load_threshold_train_p90_kW'])}\\\\text{{ kW}}$) MAE | Giờ Cao Điểm RMSE | Giờ Bình Thường MAE | Giờ Bình Thường RMSE | Tổng Thể MAE |
|:---|:---:|:---:|:---:|:---:|:---:|
{hl_table_md}

### Thảo Luận:
- Trong các khung giờ bình thường, {meta['validation_selected_winner']} đạt MAE {f4(hl_df[hl_df['model_name'] == meta['validation_selected_winner']]['normal_load_mae'].iloc[0])} kW.
- Khi phụ tải vượt ngưỡng P90, sai số MAE tăng lên {f4(hl_df[hl_df['model_name'] == meta['validation_selected_winner']]['high_load_mae'].iloc[0])} kW.
- *Giả thuyết phân tích*: Sự gia tăng sai số ở miền tải cao phản ánh tính bất định lớn hơn trong các giai đoạn hoạt động ngẫu nhiên của các thiết bị công suất cao.

---

## 9. Cơ Sở Lý Thuyết Của Các Mô Hình Nghiên Cứu

1. **Persistence (Naive Lag-1)**: Giả định $\\hat{{y}}_s = x_{{s-1}}$. Đây là mô hình cơ sở trong dự báo chuỗi thời gian, phản ánh tính trễ ngắn hạn.
2. **Seasonal Naive (Lag-24)**: Giả định $\\hat{{y}}_s = x_{{s-24}}$. Phản ánh tính lặp lại chu kỳ 24 giờ.
3. **Linear Regression (OLS)**: Ước lượng hồi quy tuyến tính cổ điển $\\hat{{y}}_s = w_0 + \\sum w_i X_{{s,i}}$ tối thiểu hóa tổng bình phương phần dư (MSE).
4. **Ridge Regression ($L_2$ Regularization)**: Bổ sung ràng buộc độ lớn hệ số $\\alpha \\sum w_i^2$ vào hàm mất mát, giúp kiểm soát hiện tượng đa cộng tuyến giữa các biến trễ liên tiếp.
5. **Random Forest Regressor**: Mô hình kết hợp biểu quyết (Bagging) từ nhiều cây quyết định độc lập trên các tập thuộc tính ngẫu nhiên, giúp kiểm soát phương sai của mô hình.
6. **Gradient Boosting Regressor**: Thuật toán tăng cường xây dựng tuần tự các cây quyết định, trong đó mỗi cây mới được tối ưu để khớp với phần dư từ các cây trước đó.
7. **XGBoost (Extreme Gradient Boosting)**: Cải tiến Gradient Boosting với xấp xỉ chuỗi Taylor bậc hai của hàm mục tiêu, kết hợp điều chuẩn độ phức tạp của cây và tối ưu hóa tính toán trên phần cứng.

---

## 10. Bộ Câu Hỏi & Trả Lời Bảo Vệ Đồ Án (Defense Q&A)

### Q1: Tại sao phải sử dụng mô hình trễ 1 giờ ($[s, s+1\\text{{h}})$ dự báo tại $s$ bằng dữ liệu $< s$) thay vì dùng các thông số cùng giờ?
**Trả lời**: Trong bài toán dự báo phụ tải, tại đầu khung giờ $s$, người vận hành cần biết trước phụ tải trong 1 giờ tới. Dữ liệu đo đạc của giờ $s$ chỉ được ghi nhận đầy đủ sau khi giờ đó đã kết thúc. Nếu đưa dữ liệu cùng giờ vào làm đầu vào thì sẽ vi phạm nguyên tắc nhân quả (Data Leakage) và khiến mô hình mất khả năng ứng dụng thực tế.

### Q2: Tại sao mô hình Persistence lại cho kết quả tốt hơn Seasonal Naive?
**Trả lời**: Ở quy mô một hộ gia đình đơn lẻ, hành vi tiêu dùng mang tính ngẫu nhiên tương đối cao giữa các ngày khác nhau (ví dụ: giờ nấu ăn, giặt giũ có thể xê dịch giữa các ngày). Do đó, thông tin của 1 giờ vừa qua phản ánh trạng thái thiết bị đang bật/tắt tốt hơn so với thông tin của cùng giờ ngày hôm trước.

### Q3: Vì sao phải sử dụng Expanding Temporal CV thay vì K-fold ngẫu nhiên?
**Trả lời**: Dữ liệu chuỗi thời gian có tính phụ thuộc thời gian (autocorrelation). Nếu chia ngẫu nhiên (shuffling), các mẫu quá khứ sẽ được huấn luyện bằng thông tin tương lai, gây rò rỉ dữ liệu và làm sai lệch độ chính xác thực tế. Expanding Temporal CV bảo đảm dữ liệu huấn luyện luôn diễn ra trước dữ liệu kiểm định.

### Q4: Mô hình này có thể áp dụng trực tiếp để tiết kiệm điện tại Việt Nam không?
**Trả lời**: Chưa thể áp dụng trực tiếp do:
1. *Đặc thù địa phương*: Dữ liệu thu thập từ một căn hộ ở Pháp (khí hậu ôn đới, cơ cấu phụ tải mùa đông khác biệt). Tại Việt Nam (khí hậu nhiệt đới), phụ tải gia đình phụ thuộc mạnh vào nhiệt độ môi trường và điều hòa nhiệt độ mùa nóng.
2. *Cơ chế tiết kiệm*: Dự báo đóng vai trò cung cấp thông tin; việc tiết kiệm chi phí đòi hỏi phải kết hợp với cơ chế giá điện theo khung giờ (TOU), pin lưu trữ, hoặc điều khiển tự động phụ tải linh hoạt.

### Q5: Tại sao trong nghiên cứu Ablation, nhóm biến Lags & Rolling lại quan trọng hơn Calendar?
**Trả lời**: Các biến lịch chỉ mang thông tin về chu kỳ ngày/tuần trung bình, trong khi nhóm biến trễ và trượt mang thông tin tức thời về mức năng lượng vừa tiêu thụ của hộ gia đình.

### Q6: Làm thế nào chứng minh hệ thống Streamlit không hiển thị kết quả ngẫu nhiên (no mock outputs)?
**Trả lời**: Streamlit gửi yêu cầu HTTP tới FastAPI. Backend xác minh checksum của gói `deploy/bundle/`, tạo đặc trưng nhân quả, nạp mô hình `.joblib` và trả dự báo cùng mã phiên bản. Frontend không nạp mô hình hay chạy huấn luyện. Kiểm thử HTTP đối chiếu dự báo API với bảng kiểm định; mã phiên bản ngăn việc trộn dữ liệu và mô hình của hai bản phát hành. Chi tiết triển khai trong [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 11. Giới Hạn Của Nghiên Cứu

1. **Dữ liệu của 01 hộ gia đình duy nhất**: Dữ liệu mang tính hành vi cá nhân của một gia đình tại Pháp trong giai đoạn 2006-2010, không đại diện cho toàn bộ các hộ gia đình khác.
2. **Thiếu biến thời tiết ngoại sinh**: Chưa tích hợp các yếu tố nhiệt độ ngoài trời, độ ẩm, bức xạ mặt trời.
3. **Mô hình dự báo một bước**: Mô hình hoạt động theo cơ chế One-step Rolling Forecast (dự báo 1 giờ tiếp theo với điều kiện các giờ trước đã được cập nhật), không phải bài toán dự báo mở nhiều ngày liên tiếp.

---

## 12. Trích Dẫn Dữ Liệu Nguồn & Bản Quyền

- **Tập dữ liệu**: Individual Household Electric Power Consumption Data Set.
- **Tác giả**: Georges Hébrail, Alice Bérard (EDF R&D).
- **Kho lưu trữ**: UCI Machine Learning Repository (Dataset 235).
- **DOI**: [10.24432/C58885](https://doi.org/10.24432/C58885).
- **Bản quyền dữ liệu**: Creative Commons Attribution 4.0 International (CC BY 4.0).
"""

    with open(ROOT_DIR / "REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    # Generate README.md
    readme_content = f"""# Dự Báo Tiêu Thụ Điện Hộ Gia Đình Theo Giờ Tiếp Theo (Next-Hour Household Active Power Forecasting)

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests: Pytest Passing](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

> **Bài tập lớn môn học Machine Learning (Course Final Project)**  
> **Chủ đề**: Ứng dụng Học máy trong Dự báo Chuỗi Thời Gian Năng Lượng Gia Đình.  
> **Tập dữ liệu**: UCI Individual Household Electric Power Consumption (Dataset ID: 235).  
> **Môi trường thử nghiệm**: Đã kiểm thử hoạt động trên Python 3.13 (Windows).

> **Kiến trúc triển khai**: Streamlit frontend → HTTP → FastAPI backend. Hướng dẫn chạy hai service, deploy miễn phí và nâng cấp mô hình: [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 1. Giới Thiệu Đề Tài & Bối Cảnh Bài Toán

Bài toán tập trung vào việc **dự báo công suất tác dụng trung bình trong 1 giờ tiếp theo** ($y_s$, đơn vị kW) của 01 hộ gia đình cho khoảng thời gian $[s, s+1\\text{{h}})$.

### Phân biệt Công suất (kW) và Điện năng (kWh)
- **Công suất tác dụng ($P$, đơn vị kW)**: Đại lượng đo tốc độ tiêu thụ năng lượng tức thời hoặc giá trị trung bình trong một khung giờ nhất định.
- **Điện năng tiêu thụ ($E$, đơn vị kWh)**: Năng lượng tích lũy tiêu hao trong khoảng thời gian $t$ ($E = P \\times t$). Vì chu kỳ dự báo ở bài toán này cố định là 1 giờ ($t = 1$ h), công suất tác dụng trung bình $1.5$ kW trong giờ đó tương ứng với $1.5$ kWh điện năng thực tế hộ gia đình tiêu thụ.

### Nguyên Tắc Nhân Quả Nghiêm Ngặt (Strict Causality)
- Dự báo cho giờ $s$ được phát tại đúng thời điểm $s$.
- Toàn bộ vector đặc trưng đầu vào $X_s$ chỉ được phép sử dụng thông tin của các giờ trước đó ($< s$, tức các giờ $s-1, s-2, \\dots$).
- Các biến trễ baseline được tính toán trên chuỗi lịch sử đã được điền tiến có giới hạn (bounded forward fill tối đa 3 giờ).
- Mô hình hoạt động theo cơ chế **One-step Rolling Forecast**: mỗi bước dự báo một giờ tiếp theo và cập nhật dữ liệu lịch sử tuần tự, không phải dự báo mở dài hạn (day-ahead open-loop).

---

## 2. Sơ Đồ Kiến Trúc Dự Án (Project Source Map)

```text
D:\\4. Final Project\\
├── artifacts\\
│   ├── figures\\            # Biểu đồ đánh giá, phần dư, so sánh, ablation và EDA
│   ├── metadata\\           # JSON metadata chi tiết về phân chia dữ liệu và CV folds
│   ├── metrics\\            # Bảng CSV kết quả CV, Validation, Test, Ablation, High-load
│   ├── models\\             # Mô hình đã huấn luyện (.joblib)
│   └── predictions\\        # Bảng dự báo test chi tiết (actual + 7 mô hình)
├── data\\
│   ├── raw\\                # Dữ liệu gốc tải từ UCI (.zip và .txt)
│   └── processed\\          # Parquet chuỗi thời gian 1 giờ đã chuẩn hóa
├── notebooks\\
│   └── exploratory_data_analysis.ipynb  # Notebook EDA có output thực thi đầy đủ
├── scripts\\
│   ├── run_eda.py          # Script thực thi EDA và xuất notebook có kết quả thực
│   └── generate_reports.py # Script tự động xuất bảng số liệu chính xác vào tài liệu
├── src\\
│   └── household_power\\    # Package mã nguồn chính
│       ├── __init__.py
│       ├── data.py         # Tải dữ liệu, lọc nhãn 60 phút, tổng hợp chuỗi giờ
│       ├── features.py     # Đặc trưng trễ, thống kê trượt, lịch tuần hoàn
│       ├── splits.py       # Phân chia thời gian (70/15/15) & Expanding CV folds
│       ├── models.py       # Định nghĩa Baselines, Scikit-learn Pipeline, XGBoost
│       ├── evaluation.py   # Tính toán MAE, RMSE, R2, Common mask, High-load error
│       ├── plots.py        # Xuất biểu đồ matplotlib chất lượng cao
│       └── pipeline.py     # Điều phối toàn bộ luồng huấn luyện và đánh giá
├── tests\\                  # Bộ kiểm thử tự động toàn diện
├── app.py                  # Giao diện web Streamlit tương tác
├── pyproject.toml          # Cấu hình đóng gói package (tested Python >=3.13)
├── pytest.ini             # Cấu hình thực thi kiểm thử
├── requirements.txt        # Danh sách thư viện và phiên bản chính xác
├── run_pipeline.py         # Điểm khởi chạy CLI đường ống huấn luyện
├── start_demo.bat          # File chạy nhanh ứng dụng giao diện (Windows Batch)
├── start_demo.ps1          # File chạy nhanh ứng dụng giao diện (PowerShell)
├── README.md               # Hướng dẫn cài đặt và sử dụng (tài liệu này)
└── REPORT.md               # Báo cáo chuyên môn chi tiết kết quả nghiên cứu
```

---

## 3. Hướng Dẫn Cài Đặt & Thực Thi Trên Windows

Toàn bộ quy trình được thiết kế chạy độc lập trong môi trường ảo `.venv` cục bộ trên hệ điều hành Windows.

### Bước 1: Khởi tạo Môi trường Ảo (Virtual Environment)
Mở PowerShell tại thư mục gốc của dự án (`D:\\4. Final Project`):

```powershell
python -m venv .venv
```

### Bước 2: Cài đặt Thư viện Phụ thuộc
Cài đặt package ở chế độ editable cùng toàn bộ các thư viện được liệt kê trong `requirements.txt`:

```powershell
.\\.venv\\Scripts\\pip.exe install -r requirements.txt
.\\.venv\\Scripts\\pip.exe install -e . --no-deps
```

Kiểm tra tính tương thích của các gói cài đặt:
```powershell
.\\.venv\\Scripts\\pip.exe check
```
*(Xác nhận: `No broken requirements found`)*

---

## 4. Các Lệnh Thực Thi Chính (Pipeline & Ứng Dụng)

### 1. Chạy Đường Ống Huấn Luyện & Đánh Giá Đầy Đủ
Lệnh sau sẽ tự động tải dữ liệu gốc từ UCI (nếu chưa có), tiền xử lý, chạy kiểm định chéo mở rộng (Expanding CV), thẩm định siêu tham số, chọn mô hình tối ưu trên tập Validation, tái huấn luyện và đánh giá trên tập Test:

```powershell
.\\.venv\\Scripts\\python.exe run_pipeline.py
```

### 2. Chạy Phân Tích Dữ Liệu Khám Phá (EDA) & Thực Thi Notebook
Tạo các biểu đồ phân tích phân phối, thiếu khuyết và xuất file Jupyter Notebook có lưu đầy đủ kết quả thực thi:

```powershell
.\\.venv\\Scripts\\python.exe scripts/run_eda.py
```

### 3. Khởi Chạy Backend FastAPI Và Frontend Streamlit
Xuất gói phục vụ từ mô hình đã huấn luyện (không train lại). Sau đó khởi chạy hai service cùng lúc:

```powershell
.\\.venv\\Scripts\\python.exe scripts/export_serving_bundle.py
.\\.venv\\Scripts\\python.exe scripts/run_local.py
```
*Hoặc nhấp đúp `start_demo.bat`. Frontend: http://127.0.0.1:8501; tài liệu API: http://127.0.0.1:8000/docs. Cấu hình cloud và cập nhật phiên bản tại [DEPLOYMENT.md](DEPLOYMENT.md).*

### 4. Chạy Bộ Kiểm Thử Tự Động (Pytest Suite)
Thực thi toàn bộ bộ kiểm thử tự động:

```powershell
.\\.venv\\Scripts\\pytest.exe -v
```

---

## 5. Tóm Tắt Kết Quả Kiểm Định (Held-Out Test Set)

Tất cả 7 mô hình được đánh giá trên cùng một mặt nạ thời gian chung (**Common Mask**: {tmask['common_valid_hours']:,} giờ kiểm định):

| STT | Mô Hình | MAE (kW) | RMSE (kW) | $R^2$ Score | Cải thiện vs Persistence | Cải thiện vs Seasonal Naive | Thời gian Fit (s) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
{readme_test_table_md}

*Ghi chú*: Mô hình **{meta['validation_selected_winner']}** được chọn làm mô hình chiến thắng căn cứ vào MAE nhỏ nhất trên tập **Validation** ({f4(val_df[val_df['model_name'] == meta['validation_selected_winner']]['mae'].iloc[0])} kW). Quá trình kiểm định được thực hiện lặp lại phục vụ mục đích thẩm định kỹ thuật mã nguồn mà không điều chỉnh siêu tham số theo tập test.

---

## 6. Chế Độ Chạy Ngoại Tuyến (Offline Reproduction)

Sau khi dữ liệu gốc đã được tải về lần đầu tại `data/raw/` và xử lý tại `data/processed/hourly_power.parquet`:
- Huấn luyện, EDA và demo hai service có thể hoạt động ngoại tuyến sau khi đã cài thư viện và tải dữ liệu. Demo local vẫn gọi HTTP giữa Streamlit và FastAPI qua loopback. Bản cloud cần kết nối Internet.

---

## 7. Trích Dẫn Dữ Liệu Nguồn & Bản Quyền

- **Tên tập dữ liệu**: Individual Household Electric Power Consumption.
- **Tác giả thu thập**: Georges Hébrail, Alice Bérard (EDF R&D, Clamart, France).
- **Kho lưu trữ**: UCI Machine Learning Repository (Dataset ID: 235).
- **DOI định danh**: [10.24432/C58885](https://doi.org/10.24432/C58885).
- **Bản quyền tập dữ liệu**: Creative Commons Attribution 4.0 International (CC BY 4.0).
"""

    with open(ROOT_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("Reports regenerated successfully from artifacts!")


if __name__ == "__main__":
    generate_reports()
