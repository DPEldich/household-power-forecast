# Dự Báo Tiêu Thụ Điện Hộ Gia Đình Theo Giờ Tiếp Theo (Next-Hour Household Active Power Forecasting)

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

Bài toán tập trung vào việc **dự báo công suất tác dụng trung bình trong 1 giờ tiếp theo** ($y_s$, đơn vị kW) của 01 hộ gia đình cho khoảng thời gian $[s, s+1\text{h})$.

### Phân biệt Công suất (kW) và Điện năng (kWh)
- **Công suất tác dụng ($P$, đơn vị kW)**: Đại lượng đo tốc độ tiêu thụ năng lượng tức thời hoặc giá trị trung bình trong một khung giờ nhất định.
- **Điện năng tiêu thụ ($E$, đơn vị kWh)**: Năng lượng tích lũy tiêu hao trong khoảng thời gian $t$ ($E = P \times t$). Vì chu kỳ dự báo ở bài toán này cố định là 1 giờ ($t = 1$ h), công suất tác dụng trung bình $1.5$ kW trong giờ đó tương ứng với $1.5$ kWh điện năng thực tế hộ gia đình tiêu thụ.

### Nguyên Tắc Nhân Quả Nghiêm Ngặt (Strict Causality)
- Dự báo cho giờ $s$ được phát tại đúng thời điểm $s$.
- Toàn bộ vector đặc trưng đầu vào $X_s$ chỉ được phép sử dụng thông tin của các giờ trước đó ($< s$, tức các giờ $s-1, s-2, \dots$).
- Các biến trễ baseline được tính toán trên chuỗi lịch sử đã được điền tiến có giới hạn (bounded forward fill tối đa 3 giờ).
- Mô hình hoạt động theo cơ chế **One-step Rolling Forecast**: mỗi bước dự báo một giờ tiếp theo và cập nhật dữ liệu lịch sử tuần tự, không phải dự báo mở dài hạn (day-ahead open-loop).

---

## 2. Sơ Đồ Kiến Trúc Dự Án (Project Source Map)

```text
D:\4. Final Project\
├── artifacts\
│   ├── figures\            # Biểu đồ đánh giá, phần dư, so sánh, ablation và EDA
│   ├── metadata\           # JSON metadata chi tiết về phân chia dữ liệu và CV folds
│   ├── metrics\            # Bảng CSV kết quả CV, Validation, Test, Ablation, High-load
│   ├── models\             # Mô hình đã huấn luyện (.joblib)
│   └── predictions\        # Bảng dự báo test chi tiết (actual + 7 mô hình)
├── data\
│   ├── raw\                # Dữ liệu gốc tải từ UCI (.zip và .txt)
│   └── processed\          # Parquet chuỗi thời gian 1 giờ đã chuẩn hóa
├── notebooks\
│   └── exploratory_data_analysis.ipynb  # Notebook EDA có output thực thi đầy đủ
├── scripts\
│   ├── run_eda.py          # Script thực thi EDA và xuất notebook có kết quả thực
│   └── generate_reports.py # Script tự động xuất bảng số liệu chính xác vào tài liệu
├── src\
│   └── household_power\    # Package mã nguồn chính
│       ├── __init__.py
│       ├── data.py         # Tải dữ liệu, lọc nhãn 60 phút, tổng hợp chuỗi giờ
│       ├── features.py     # Đặc trưng trễ, thống kê trượt, lịch tuần hoàn
│       ├── splits.py       # Phân chia thời gian (70/15/15) & Expanding CV folds
│       ├── models.py       # Định nghĩa Baselines, Scikit-learn Pipeline, XGBoost
│       ├── evaluation.py   # Tính toán MAE, RMSE, R2, Common mask, High-load error
│       ├── plots.py        # Xuất biểu đồ matplotlib chất lượng cao
│       └── pipeline.py     # Điều phối toàn bộ luồng huấn luyện và đánh giá
├── tests\                  # Bộ kiểm thử tự động toàn diện
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
Mở PowerShell tại thư mục gốc của dự án (`D:\4. Final Project`):

```powershell
python -m venv .venv
```

### Bước 2: Cài đặt Thư viện Phụ thuộc
Cài đặt package ở chế độ editable cùng toàn bộ các thư viện được liệt kê trong `requirements.txt`:

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\pip.exe install -e . --no-deps
```

Kiểm tra tính tương thích của các gói cài đặt:
```powershell
.\.venv\Scripts\pip.exe check
```
*(Xác nhận: `No broken requirements found`)*

---

## 4. Các Lệnh Thực Thi Chính (Pipeline & Ứng Dụng)

### 1. Chạy Đường Ống Huấn Luyện & Đánh Giá Đầy Đủ
Lệnh sau sẽ tự động tải dữ liệu gốc từ UCI (nếu chưa có), tiền xử lý, chạy kiểm định chéo mở rộng (Expanding CV), thẩm định siêu tham số, chọn mô hình tối ưu trên tập Validation, tái huấn luyện và đánh giá trên tập Test:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

### 2. Chạy Phân Tích Dữ Liệu Khám Phá (EDA) & Thực Thi Notebook
Tạo các biểu đồ phân tích phân phối, thiếu khuyết và xuất file Jupyter Notebook có lưu đầy đủ kết quả thực thi:

```powershell
.\.venv\Scripts\python.exe scripts/run_eda.py
```

### 3. Khởi Chạy Backend FastAPI Và Frontend Streamlit
Xuất gói phục vụ từ mô hình đã huấn luyện (không train lại). Sau đó khởi chạy hai service cùng lúc:

```powershell
.\.venv\Scripts\python.exe scripts/export_serving_bundle.py
.\.venv\Scripts\python.exe scripts/run_local.py
```
*Hoặc nhấp đúp `start_demo.bat`. Frontend: http://127.0.0.1:8501; tài liệu API: http://127.0.0.1:8000/docs. Cấu hình cloud và cập nhật phiên bản tại [DEPLOYMENT.md](DEPLOYMENT.md).*

### 4. Chạy Bộ Kiểm Thử Tự Động (Pytest Suite)
Thực thi toàn bộ bộ kiểm thử tự động:

```powershell
.\.venv\Scripts\pytest.exe -v
```

---

## 5. Tóm Tắt Kết Quả Kiểm Định (Held-Out Test Set)

Tất cả 7 mô hình được đánh giá trên cùng một mặt nạ thời gian chung (**Common Mask**: 4,900 giờ kiểm định):

| STT | Mô Hình | MAE (kW) | RMSE (kW) | $R^2$ Score | Cải thiện vs Persistence | Cải thiện vs Seasonal Naive | Thời gian Fit (s) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Persistence | 0.3892 | 0.5885 | 0.3155 | +0.00% | +23.68% | 0.00 |
| 2 | Seasonal Naive | 0.5100 | 0.7601 | -0.1418 | -31.03% | +0.00% | 0.00 |
| 3 | Linear Regression | 0.3567 | 0.5022 | 0.5014 | +8.34% | +30.04% | 0.07 |
| 4 | Ridge | 0.3567 | 0.5022 | 0.5015 | +8.34% | +30.04% | 0.08 |
| 5 | Random Forest | 0.3311 | 0.4747 | 0.5547 | +14.93% | +35.08% | 3.72 |
| 6 | Gradient Boosting | 0.3279 | 0.4707 | 0.5622 | +15.74% | +35.70% | 13.21 |
| 7 | **XGBoost** | **0.3242** | **0.4666** | **0.5698** | **+16.70%** | **+36.42%** | **0.31** |

*Ghi chú*: Mô hình **XGBoost** được chọn làm mô hình chiến thắng căn cứ vào MAE nhỏ nhất trên tập **Validation** (0.3706 kW). Quá trình kiểm định được thực hiện lặp lại phục vụ mục đích thẩm định kỹ thuật mã nguồn mà không điều chỉnh siêu tham số theo tập test.

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


---

## 8. Phụ Lục ML-03: Thử Nghiệm Mở Rộng Động Lực Phụ Tải & Dự Báo Khoảng Bất Định

Phiên bản mở rộng ML-03 bổ sung phân tích động lực phụ tải (load dynamics) và ước lượng dải bất định phân vị P10/P50/P90 nhằm định lượng rủi ro trong dự báo tiêu thụ điện gia đình.

> **Lưu ý phương pháp luận quan trọng**: Việc đánh giá trên tập kiểm định Held-Out Test (15% cuối) là việc tái sử dụng tập chuẩn đối chiếu lịch sử (reused historical benchmark) đã có từ ML-01, hoàn toàn không phải là bằng chứng kiểm định độc lập mới chưa từng thấy (unseen independent evidence). Toàn bộ thử nghiệm ML-03 chỉ được thực thi và kiểm thử tại môi trường cục bộ (local execution); không thực hiện xuất bản lên hạ tầng đám mây (no cloud publication performed).

### 8.1 Đặc trưng động lực phụ tải quá khứ (Causal Load Dynamics)
Bổ sung 6 đặc trưng quá khứ nghiêm ngặt (chỉ dùng dữ liệu $< s$):
- `diff_1h`: Tốc độ biến thiên công suất 1 giờ gần nhất ($x_{s-1} - x_{s-2}$).
- `diff_24h`: Tốc độ biến thiên công suất so với cùng giờ hôm trước ($x_{s-1} - x_{s-25}$).
- `rolling_max_3`, `rolling_max_24`: Mức công suất đỉnh trong 3 giờ và 24 giờ qua.
- `trend_3h`: Độ lệch xu hướng giữa mức công suất tức thời $x_{s-1}$ và mức trung bình 3 giờ trước.
- `missing_lag1`: Chỉ báo khuyết thiếu dữ liệu tại bước trễ gần nhất.

### 8.2 Giao thức kiểm định trung thực & Bảo toàn mô hình Baseline
- **Expanding CV (3 folds trên 70% Train)**: Ngưỡng phụ tải cao P90 tính riêng theo từng fold train là $2.4883$, $2.5970$, và $2.4747$ kW. Ứng viên `enhanced_features` giảm MAE phụ tải cao (1.0426 vs 1.0469 kW) và giữ MAE tổng thể (0.3737 vs 0.3760 kW), được chọn tại vòng CV.
- **Xác nhận trên tập Validation (15% theo thời gian)**: Ngưỡng P90 tính trên 70% Train là **2.3800 kW**. Mặc dù MAE tổng thể cải thiện (0.3696 vs 0.3706 kW), nhưng MAE phụ tải cao tăng nhẹ (0.8409 vs 0.8360 kW).
- **Quyết định bảo toàn**: Tuân thủ giao thức khoa học đã cố định trước (`protocol.json` và `selection_frozen.json`), hệ thống từ chối thăng cấp ứng viên và bảo toàn mô hình Baseline XGBoost cho dự báo điểm.

### 8.3 Dự báo khoảng phân vị P10 – P50 – P90
- Huấn luyện 3 mô hình XGBoost với hàm mất mát `reg:quantileerror` tại $\alpha \in [0.1, 0.5, 0.9]$.
- Thuật toán `repair_and_clip_quantiles` bảo đảm không có công suất âm và triệt tiêu lỗi bắt chéo phân vị ($0 \le P10 \le P50 \le P90$).
- Mô hình điểm sử dụng hàm mất mát sai số bình phương `reg:squarederror` (ước lượng kỳ vọng có điều kiện $E[Y|X]$), trong khi P50 tối ưu hóa Pinball Loss ($\alpha = 0.5$) để ước lượng trung vị có điều kiện $\text{Median}[Y|X]$. Hai mục tiêu thống kê này có bản chất toán học khác nhau.

#### Bảng số liệu bất định phân vị (Validation & Test Set)

| Split | Phân khúc (Subset) | Số mẫu (Count) | Độ phủ (%) (Coverage) | Dưới P10 (%) | Vượt P90 (%) | Độ rộng TB (kW) | Pinball P10 | Pinball P50 | Pinball P90 | Pinball TB (kW) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| validation | overall | 5017 | 88.18% | 4.09% | 7.73% | 1.5419 | 0.0759 | 0.1995 | 0.1195 | 0.1316 |
| validation | high_load | 583 | 44.77% | 0.00% | 55.23% | 1.9478 | 0.2099 | 0.6268 | 0.2928 | 0.3765 |
| validation | normal_load | 4434 | 93.89% | 4.62% | 1.49% | 1.4885 | 0.0583 | 0.1433 | 0.0967 | 0.0994 |
| test | overall | 4900 | 90.49% | 5.47% | 4.04% | 1.4623 | 0.0579 | 0.1633 | 0.1099 | 0.1104 |
| test | high_load | 226 | 38.05% | 0.00% | 61.95% | 1.8995 | 0.2130 | 0.6550 | 0.3219 | 0.3966 |
| test | normal_load | 4674 | 93.03% | 5.73% | 1.24% | 1.4412 | 0.0504 | 0.1395 | 0.0997 | 0.0965 |

- **Giả thuyết phân tích quan sát**: Tại phân khúc phụ tải cao, độ phủ thực nghiệm giảm sâu (38.05% trên Test và 44.77% trên Validation), với trên 55–61% số giờ đỉnh nhọn thực tế vượt ra ngoài biên trên P90. Đây có thể là hệ quả của việc phân phối tải có đuôi rất dày khi các thiết bị công suất lớn bật đột ngột, khiến mô hình phân vị toàn dải bị co cụm về phía trung vị. Đây là một giả thuyết phân tích quan sát, không phải là kết luận nhân quả đã được chứng minh thực nghiệm.
- **Cảnh báo giới hạn**: Độ phủ thực nghiệm tổng thể cao (90.49%) không đồng nghĩa với bảo đảm an toàn ở các giờ đỉnh nhọn hay bảo đảm hiệu chuẩn hoàn hảo (calibrated uncertainty). Không sử dụng khoảng dự báo này cho các bài toán an toàn cắt tải mà không có hệ số đệm mở rộng.

### 8.4 Gói phát hành serving v2 & API mở rộng
- Đóng gói có mã băm SHA256 cho toàn bộ mô hình và bảng kết quả tại `deploy/bundle/`.
- Endpoint `POST /v1/predict_interval` hỗ trợ suy luận điểm và dải phân vị.
- Tính trùng khớp số học giữa `historical_replay` và `provided_history` đòi hỏi ngữ cảnh lịch sử tương đương (ví dụ 336 giờ cung cấp ngữ cảnh đầy đủ cho cửa sổ 168 giờ và việc bù khuyết giới hạn); ca kiểm thử tự động `test_predict_interval_provided_history_parity` là bằng chứng kiểm chứng cụ thể cho dòng kiểm tra đã chọn.
- Chi tiết báo cáo thực nghiệm xem tại [ML03_RESULTS.md](ML03_RESULTS.md).
