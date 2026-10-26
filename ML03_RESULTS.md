# BÁO CÁO THỰC THI THỬ NGHIỆM ML-03: ĐỘNG LỰC PHỤ TẢI & ĐÁNH GIÁ ĐỘ BẤT ĐỊNH

> **Mã hội thoại (Conversation ID)**: `38f8b4cf-c23d-4afa-96a7-f49782408c6a`  
> **Thời điểm hoàn thành**: 2026-10-26  
> **Môi trường thực thi**: Python 3.13.15, XGBoost 3.4.1, scikit-learn, FastAPI, Streamlit  
> **Trạng thái kết quả**: Hoàn thành toàn diện mã nguồn, thực nghiệm thực tế trên dữ liệu UCI, tích hợp bundle serving có xác thực checksum, kiểm thử hồi quy 100% đạt, giao diện tiếng Việt.

---

## 1. Danh Sách Tệp Đã Thay Đổi & Tạo Mới

### Mã nguồn chính (`src/household_power/`)
- `src/household_power/uncertainty.py` *(Tạo mới)*: Thư viện thuần túy toán học / NumPy tính hàm mất mát phân vị `pinball_loss`, hàm sửa lỗi bắt chéo phân vị và cắt ngưỡng không âm `repair_and_clip_quantiles`, và hàm đánh giá bất định `evaluate_uncertainty` với xác thực mảng 1D, cấm NaN/Inf, kiểm tra tính đơn điệu của phân vị và xử lý an toàn tập phụ tải cao rỗng.
- `src/household_power/features.py` *(Cập nhật)*: Bổ sung 6 đặc trưng động lực phụ tải quá khứ nghiêm ngặt (`diff_1h`, `diff_24h`, `rolling_max_3`, `rolling_max_24`, `trend_3h`, `missing_lag1`) trong hàm `build_extended_features`, bảo toàn hợp đồng 20 đặc trưng cơ sở của ML-01.
- `src/household_power/experiments.py` *(Tạo mới)*: Bộ điều phối thực nghiệm ML-03, ghi nhận `protocol.json` trước khi chạy CV, liên kết cấu hình XGBoost cố định từ metadata ML-01, thực thi Expanding Temporal CV 3 folds, xác nhận trên Validation, ghi nhận `selection_frozen.json` trước khi chấm điểm test, huấn luyện các mô hình phân vị P10/P50/P90, căn chỉnh dự báo test baseline bằng timestamp, lưu trữ toàn bộ metrics, predictions và biểu đồ phân tích.
- `src/household_power/serving.py` *(Cập nhật)*: Mở rộng `ForecastService` hỗ trợ schema v2 tuỳ chọn, xác thực checksum toàn bộ mô hình và bảng kết quả trước khi nạp, kiểm tra thứ tự đặc trưng khớp với metadata và `feature_names_in_` của estimator, cung cấp phương thức `predict_interval` hỗ trợ cả historical replay và provided history.
- `src/household_power/api.py` *(Cập nhật)*: Bổ sung endpoint `POST /v1/predict_interval` với mô hình pydantic `IntervalPredictionRequest`, kiểm tra mã phiên bản mô hình, chuẩn hóa timestamp theo giờ và bảo toàn nguyên vẹn endpoint `POST /v1/predict` cho 7 mô hình baseline.

### Kịch bản CLI (`scripts/`)
- `scripts/run_experiments.py` *(Tạo mới)*: Kịch bản dòng lệnh tái hiện độc lập toàn bộ quy trình thực nghiệm ML-03 trên dữ liệu UCI thực tế.
- `scripts/export_serving_bundle.py` *(Cập nhật)*: Đóng gói release serving gồm cả 7 mô hình baseline và phần mở rộng thí nghiệm v2 có mã băm SHA256, tự động kiểm tra đối chiếu (assert_allclose) toàn bộ dự báo test của mô hình điểm và phân vị trước khi cho phép xuất bản.

### Kiểm thử (`tests/`)
- `tests/test_ml03_causality.py` *(Tạo mới)*: Kiểm tra tính bất biến nhân quả tuyệt đối của 26 đặc trưng mở rộng khi làm nhiễu dữ liệu hiện tại và tương lai, đồng thời kiểm tra từ chối chuỗi thời gian không liên tục.
- `tests/test_ml03_selection_guard.py` *(Tạo mới)*: Kiểm tra trực tiếp các hàm logic sản xuất `evaluate_candidate_cv`, `confirm_validation_candidate`, `compute_high_load_threshold`, `align_baseline_test_predictions` với các trường hợp biên (ngưỡng 1% overall MAE, số fold không thoái hóa 1 vs 2, fallback validation).
- `tests/test_ml03_uncertainty_metrics.py` *(Tạo mới)*: Kiểm tra pinball loss giải tích, thuật toán sửa lỗi quantile crossing, xử lý mảng phụ tải cao rỗng và từ chối đầu vào NaN/Inf/2D.
- `tests/test_ml03_train_isolation.py` *(Tạo mới)*: Kiểm tra ngưỡng phụ tải cao và bộ điền khuyết SimpleImputer chỉ học từ tập huấn luyện (Train-only).
- `tests/test_ml03_parity.py` *(Tạo mới)*: Kiểm tra đối chiếu toàn bộ 4.900 dòng dự báo kiểm định giữa file CSV và kết quả suy luận từ các file mô hình `.joblib`.
- `tests/test_ml03_bundle_security.py` *(Tạo mới)*: Kiểm tra cơ chế fail-closed khi checksum bị sai lệch, thiếu vai trò mô hình bắt buộc, thiếu metric, thứ tự đặc trưng bị đảo ngược, hoặc đường dẫn thoát khỏi thư mục bundle; đồng thời kiểm tra bundle legacy không có thí nghiệm vẫn hoạt động bình thường.
- `tests/test_ml03_api_and_history.py` *(Tạo mới)*: Kiểm tra tính tương thích ngược của `/v1/predict`, chức năng `/v1/predict_interval`, sự trùng khớp số học tuyệt đối giữa provided history và historical replay, và các ràng buộc mã lỗi 409, 422.
- `tests/test_ml03_streamlit_apptest.py` *(Tạo mới)*: Kiểm tra tương tác Streamlit frontend với backend FastAPI thật thông qua Streamlit AppTest cho nút suy luận khoảng và toggle dải bóng mờ.

### Giao diện và Tài liệu
- `app.py` *(Cập nhật)*: Mở rộng 6 tab giao diện tiếng Việt hiển thị thông tin đối chiếu CV/Validation, cảnh báo trung thực về tỷ lệ bao phủ phụ tải cao, toggle dải bóng mờ P10–P90, nút suy luận trực tiếp `/v1/predict_interval`, phân tích phương pháp luận và giới hạn.
- `README.md` *(Cập nhật)*: Bổ sung phụ lục Mục 8 về thử nghiệm ML-03.
- `REPORT.md` *(Cập nhật)*: Bổ sung phụ lục Mục 13 về báo cáo kỹ thuật chuyên sâu ML-03.
- `DEPLOYMENT.md` *(Cập nhật)*: Bổ sung phụ lục Mục 9 về thông số kỹ thuật bundle v2 và API interval.
- `artifacts/experimental/ml03/` *(Tạo mới)*: Thư mục chứa toàn bộ protocol, metadata, metrics CSV, predictions CSV, models joblib và figures PNG.

---

## 2. Giao Thức Thực Nghiệm & Danh Sách Ứng Viên

### 2.1 Cấu hình mô hình cố định
- Thuật toán: XGBoost Regressor
- Siêu tham số cố định (trích xuất từ `split_metadata.json` của ML-01):
  `{"n_estimators": 100, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "random_state": 42, "n_jobs": 4}`

### 2.2 Danh sách ứng viên khai báo trước (Declared Candidates)
1. **`baseline`**: Cấu hình XGBoost cố định, tập 20 đặc trưng cơ sở ML-01, không gán trọng số mẫu.
2. **`enhanced_features`**: Cấu hình XGBoost cố định, tập 26 đặc trưng mở rộng (thêm 6 đặc trưng động lực phụ tải), không gán trọng số mẫu.
3. **`enhanced_weighted`**: Cấu hình XGBoost cố định, tập 26 đặc trưng mở rộng, gán trọng số mẫu nhẹ ($w = 1.5$) cho các mẫu huấn luyện có $y \ge \text{ngưỡng phụ tải cao P90}$ tính riêng trên fold train.

### 2.3 Quy tắc lựa chọn ứng viên nghiêm ngặt
- **Quy tắc CV (Expanding Temporal CV 3 folds trên 70% Train đầu)**:
  1. MAE tổng thể trung bình $\le \text{Baseline} \times 1.01$ (không suy giảm quá 1% sai số tổng thể).
  2. Ít nhất 2 trong 3 fold có MAE phụ tải cao $\le \text{Baseline}$ (tính theo ngưỡng P90 riêng của fold train đó).
  3. Lựa chọn ứng viên có MAE phụ tải cao trung bình nhỏ nhất trong các ứng viên hợp lệ.
- **Quy tắc xác nhận trên tập Validation (15% độc lập theo thời gian)**:
  1. MAE tổng thể trên Validation $\le \text{Baseline Validation} \times 1.01$.
  2. MAE phụ tải cao trên Validation $\le \text{Baseline Validation High-Load}$.
  3. **Nguyên tắc bảo toàn**: Nếu không đáp ứng đồng thời cả 2 điều kiện trên, hệ thống bảo toàn mô hình điểm Baseline và báo cáo trung thực.

---

## 3. Lệnh Thực Thi Thực Tế & Kết Quả Kiểm Thử

### 3.1 Thực thi thực nghiệm trên dữ liệu thật
```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py
```
- Lệnh chạy thành công, hoàn thành trong khoảng 9 giây trên dữ liệu 2.075.259 phút đo đạc thực tế của UCI (tổng hợp thành 34.589 giờ liên tục).

### 3.2 Đóng gói serving bundle
```powershell
.\.venv\Scripts\python.exe scripts\export_serving_bundle.py
```
- Kết quả xuất bản: `Exported uci-hourly-9c37e566d5f6e848: 7 baseline models verified, experimental v2 models verified, 4900 test hours`.
- Toàn bộ 7 mô hình baseline và 4 mô hình experimental v2 (`selected_point_model`, `quantile_p10`, `quantile_p50`, `quantile_p90`) được thẩm định khớp tuyệt đối với file CSV dự báo trước khi xuất bản.

### 3.3 Toàn bộ kiểm thử tự động (Full Pytest Suite)
```powershell
.\.venv\Scripts\python.exe -m pytest
```
- **Kết quả**: **80 / 80 passed (100%)**, 3 warnings không liên quan (Starlette deprecation và FigureCanvas non-interactive).
- **Kiểm tra phụ thuộc (pip check)**:
```powershell
.\.venv\Scripts\python.exe -m pip check
```
- **Kết quả**: `No broken requirements found.`

---

## 4. Dữ Liệu Thực Nghiệm Chi Tiết (Evidence)

### 4.1 Kết quả Cross-Validation (Expanding CV 3 Folds)
Ngưỡng phụ tải cao P90 tính riêng theo từng Fold Train:
- Fold 1 (5.940 mẫu valid): P90 = 2.4883 kW
- Fold 2 (11.954 mẫu valid): P90 = 2.5970 kW
- Fold 3 (17.964 mẫu valid): P90 = 2.4747 kW

| Ứng Viên | MAE Tổng Thể (kW) | Tỷ lệ vs Baseline | MAE Phụ Tải Cao (kW) | Số Fold Không Thoái Hóa | Đạt Điều Kiện CV |
|:---|:---:|:---:|:---:|:---:|:---:|
| `baseline` | 0.3760 | 1.0000 | 1.0469 | 3 / 3 | - |
| `enhanced_features` | **0.3737** | **0.9939** (<= 1.01) | **1.0426** (< baseline) | **2 / 3** | **ĐẠT (Eligible)** |
| `enhanced_weighted` | 0.3803 | 1.0115 (> 1.01) | 0.9752 | 3 / 3 | KHÔNG (Vi phạm 1% tổng thể) |

*Kết luận vòng CV*: Ứng viên `enhanced_features` được chọn tại vòng CV vì cải thiện MAE phụ tải cao (1.0426 vs 1.0469 kW) và đáp ứng ràng buộc MAE tổng thể (cải thiện 0.61%).

### 4.2 Kết quả Xác Nhận trên Tập Validation (Confirmation Split - 15% thời gian)
- Số giờ đánh giá chung (Common Mask): 5.017 giờ (trên tổng số 5.163 giờ lịch).
- Ngưỡng phụ tải cao P90 (tính trên 100% tập Train 70%): **2.3800 kW**.

| Mô Hình | MAE Tổng Thể (kW) | RMSE Tổng Thể (kW) | MAE Phụ Tải Cao (kW) | Đánh Giá Ràng Buộc | Quyết Định |
|:---|:---:|:---:|:---:|:---|:---:|
| `baseline` | 0.3706 | 0.5368 | **0.8360** | Chuẩn đối chiếu | - |
| `enhanced_features` | **0.3696** | 0.5366 | **0.8409** | Overall: Đạt (0.3696 <= 0.3743)<br>High-load: **KHÔNG ĐẠT (0.8409 > 0.8360)** | **BỊ TỪ CHỐI (Reverted)** |

*Quyết định kỹ thuật trung thực*: Mặc dù `enhanced_features` cải thiện nhẹ sai số tổng thể trên Validation (từ 0.3706 xuống 0.3696 kW), nhưng sai số ở phân khúc phụ tải cao bị tăng nhẹ từ 0.8360 lên 0.8409 kW (+0.59%). Theo đúng quy tắc giao thức đã cố định trước, **hệ thống từ chối thăng cấp và bảo toàn mô hình Baseline XGBoost làm mô hình điểm**.

### 4.3 Kết quả Đối Chiếu Kỹ Thuật trên Held-Out Test (Descriptive Benchmark)
- Số giờ kiểm định chung (Common Mask): 4.900 giờ (trên tổng số 5.164 giờ lịch).
- Mô hình điểm được tái huấn luyện trên 85% dữ liệu đầu (First 85%) và đánh giá descriptive trên Test:

| Mô Hình Điểm | MAE Tổng Thể (kW) | RMSE Tổng Thể (kW) | MAE Phụ Tải Cao (kW) | MAE Tải Thường (kW) |
|:---|:---:|:---:|:---:|:---:|
| `baseline_xgboost` (ML-01) | 0.324208 | 0.466553 | 1.002815 | 0.291395 |
| `baseline` (ML-03 refit) | 0.324208 | 0.466553 | 1.002815 | 0.291395 |

*Độ khớp*: Kết quả tái lập trùng khớp tuyệt đối đến 6 chữ số thập phân với mô hình ML-01 gốc.

### 4.4 Kết quả Mô Hình Phân Vị & Đánh Giá Bất Định (P10 / P50 / P90)
Mô hình sử dụng XGBoost `reg:quantileerror` với $\alpha \in [0.1, 0.5, 0.9]$, áp dụng thuật toán cắt ngưỡng không âm và sửa lỗi crossing đơn điệu ($0 \le P10 \le P50 \le P90$).

#### Bảng số liệu bất định phân vị toàn diện (Được tạo trực tiếp từ uncertainty_metrics.csv)

| Split | Phân khúc (Subset) | Số mẫu (Count) | Độ phủ (%) (Coverage) | Dưới P10 (%) | Vượt P90 (%) | Độ rộng TB (kW) | Pinball P10 | Pinball P50 | Pinball P90 | Pinball TB (kW) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| validation | overall | 5017 | 88.18% | 4.09% | 7.73% | 1.5419 | 0.0759 | 0.1995 | 0.1195 | 0.1316 |
| validation | high_load | 583 | 44.77% | 0.00% | 55.23% | 1.9478 | 0.2099 | 0.6268 | 0.2928 | 0.3765 |
| validation | normal_load | 4434 | 93.89% | 4.62% | 1.49% | 1.4885 | 0.0583 | 0.1433 | 0.0967 | 0.0994 |
| test | overall | 4900 | 90.49% | 5.47% | 4.04% | 1.4623 | 0.0579 | 0.1633 | 0.1099 | 0.1104 |
| test | high_load | 226 | 38.05% | 0.00% | 61.95% | 1.8995 | 0.2130 | 0.6550 | 0.3219 | 0.3966 |
| test | normal_load | 4674 | 93.03% | 5.73% | 1.24% | 1.4412 | 0.0504 | 0.1395 | 0.0997 | 0.0965 |

#### Chi tiết số liệu:
- **Tập Validation (5.017 giờ)**:
  - *Tổng thể (5.017 mẫu)*: Độ phủ thực nghiệm đạt **88.18%**, tỷ lệ dưới P10: **4.09%**, tỷ lệ vượt P90: **7.73%**, độ rộng khoảng trung bình: **1.5419 kW**, pinball loss trung bình: **0.1316 kW** (P10: 0.0759, P50: 0.1995, P90: 0.1195).
  - *Phụ tải cao (583 mẫu $\ge 2.3800$ kW)*: Độ phủ thực nghiệm đạt **44.77%**, tỷ lệ vượt P90: **55.23%**, độ rộng khoảng trung bình: **1.9478 kW**, pinball loss trung bình: **0.3765 kW** (P10: 0.2099, P50: 0.6268, P90: 0.2928).
  - *Phụ tải thường (4.434 mẫu $< 2.3800$ kW)*: Độ phủ thực nghiệm đạt **93.89%**, độ rộng khoảng trung bình: **1.4885 kW**, pinball loss trung bình: **0.0994 kW** (P10: 0.0583, P50: 0.1433, P90: 0.0967).
- **Tập Held-Out Test (4.900 giờ)**:
  - *Tổng thể (4.900 mẫu)*: Độ phủ thực nghiệm đạt **90.49%** (danh định 80.00%), tỷ lệ dưới P10: **5.47%**, tỷ lệ vượt P90: **4.04%**, độ rộng khoảng trung bình: **1.4623 kW**, pinball loss trung bình: **0.1104 kW** (P10: **0.057927**, P50: **0.163263**, P90: **0.109909**).
  - *Phụ tải cao (226 mẫu $\ge 2.3800$ kW)*: Độ phủ thực nghiệm đạt **38.05%**, tỷ lệ vượt P90: **61.95%** (hơn 61% số giờ đỉnh nhọn thực tế vượt ra ngoài biên trên P90), độ rộng khoảng trung bình: **1.8995 kW**, pinball loss trung bình: **0.3966 kW** (P10: 0.2130, P50: 0.6550, P90: 0.3219).
  - *Phụ tải thường (4.674 mẫu $< 2.3800$ kW)*: Độ phủ thực nghiệm đạt **93.03%**, độ rộng khoảng trung bình: **1.4412 kW**, pinball loss trung bình: **0.0965 kW** (P10: 0.0504, P50: 0.1395, P90: 0.0997).

---

## 5. Phân Tích Kỹ Thuật & Giới Hạn Cốt Lõi

1. **Hiểu đúng về độ phủ danh định 80% (Nominal Target vs Observed Coverage)**:
   - Mức độ phủ 80% là mục tiêu lý thuyết khi tối ưu hàm mất mát phân vị $\alpha = 0.1$ và $\alpha = 0.9$.
   - Trên tổng thể, mô hình đạt độ phủ 90.49% trên Test và 88.18% trên Validation (đều vượt mục tiêu), tuy nhiên tại phân khúc phụ tải cao (High-Load), độ phủ thực nghiệm giảm sâu xuống **38.05%** trên Test và **44.77%** trên Validation.
   - **Giả thuyết phân tích quan sát**: Dữ liệu phụ tải điện hộ gia đình có thể có phân phối đuôi rất nặng khi các thiết bị công suất lớn bật đột ngột. Mô hình hồi quy phân vị toàn dải có xu hướng bị co cụm về phía vùng mật độ cao, dẫn đến việc ước lượng dưới mức đỉnh nhọn (under-estimation of extreme peaks). Đây là một giả thuyết phân tích quan sát, không phải là kết luận nhân quả đã được chứng minh thực nghiệm.
   - **Khuyến cáo**: Không bao giờ được tuyên bố rằng khoảng dự báo đã được hiệu chuẩn tin cậy cho các ứng dụng an toàn lưới điện hoặc bảo vệ quá tải. Độ phủ thực nghiệm cao ở tải thường không đồng nghĩa với bảo đảm an toàn vận hành.

2. **Phân biệt rạch ròi giữa P50 và Điểm dự báo (Point Prediction)**:
   - Mô hình điểm XGBoost được huấn luyện với mục tiêu squared-error (`reg:squarederror`), hướng tới kỳ vọng có điều kiện $E[Y|X]$.
   - Mô hình phân vị P50 được huấn luyện với hàm mất mát Pinball Loss ($\alpha = 0.5$, `reg:quantileerror`), hướng tới trung vị có điều kiện $\text{Median}[Y|X]$.
   - Đây là hai mục tiêu thống kê độc lập có bản chất toán học khác nhau; tuyệt đối không xem xét sai số MAE và MSE như những mục tiêu tương đương và không thay thế P50 cho dự báo điểm.

3. **Bản chất của tập kiểm định Held-Out Test**:
   - Tập dữ liệu 15% cuối cùng (từ 25/04/2010 đến 26/11/2010) là tập dữ liệu lịch sử đã được dùng trong ML-01.
   - Việc đánh giá trên tập này trong ML-03 hoàn toàn chỉ mang tính chất **đối chiếu kỹ thuật (reused historical benchmark)**, không phải là bằng chứng kiểm định độc lập mới chưa từng thấy (unseen independent evidence).

4. **Giới hạn quy mô dữ liệu & Phạm vi môi trường**:
   - Dữ liệu thu thập từ 01 hộ gia đình duy nhất tại Sceaux, ngoại ô Paris (Pháp) trong giai đoạn 2006–2010. Không thể khái quát hóa cho các hộ gia đình tại Việt Nam hay các vùng khí hậu khác.
   - Toàn bộ thử nghiệm ML-03 chỉ được thực thi và kiểm thử tại môi trường cục bộ (local execution); không thực hiện xuất bản lên hạ tầng đám mây (no cloud publication performed).

---

## 6. Báo Cáo Tồn Tại & Trở Ngại (Blockers)

- Không có trở ngại kỹ thuật (blocker) nào tồn tại.
- Toàn bộ 80 ca kiểm thử tự động đều đạt (100% pass).
- Codex đã nghiệm thu độc lập bản cuối; bằng chứng và giới hạn được ghi tại [ML03_REVIEW.md](ML03_REVIEW.md).
