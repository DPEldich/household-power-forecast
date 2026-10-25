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
