# BÁO CÁO KẾT THÚC HỌC PHẦN MACHINE LEARNING
## ĐỀ TÀI: DỰ BÁO CÔNG SUẤT TIÊU THỤ ĐIỆN HỘ GIA ĐÌNH THEO GIỜ TIẾP THEO (NEXT-HOUR HOUSEHOLD ACTIVE POWER FORECASTING)

---

## Tóm Tắt Đồ Án (Executive Summary)

Đề tài giải quyết bài toán dự báo chuỗi thời gian phụ tải điện gia đình: **ước lượng công suất tác dụng trung bình trong 1 giờ tiếp theo ($y_s$, đơn vị kW)** của một hộ gia đình cho khoảng thời gian $[s, s+1\text{h})$, sử dụng thông tin lịch sử diễn ra trước thời điểm $s$ ($< s$).

Nghiên cứu sử dụng tập dữ liệu đo đạc thực tế **UCI Individual Household Electric Power Consumption** (Dataset 235), gồm **2,075,259 phút đo đạc** trong giai đoạn từ tháng 12/2006 đến tháng 11/2010 (2006-12-16 đến 2010-11-26).

Quy trình nghiên cứu áp dụng các nguyên tắc về tính toàn vẹn nhân quả (Strict Causality) và kiểm định học máy chuẩn mực:
1. **Phân chia thời gian**: 168 giờ khởi động (Warmup), 70% thời gian đầu làm tập Train (chứa 3-fold Expanding Temporal Cross-Validation), 15% thời gian tiếp theo làm tập Validation (để chọn cấu hình tối ưu), và 15% thời gian cuối cùng làm tập Test.
2. **Khảo sát 7 mô hình**: 2 mô hình chuẩn cơ sở (Persistence, Seasonal Naive), 2 mô hình hồi quy tuyến tính (Linear Regression, Ridge Regression) và 3 mô hình học máy dạng cây (Random Forest, Gradient Boosting, XGBoost).
3. **Mặt nạ đánh giá chung (Common Mask)**: Toàn bộ 7 mô hình được đánh giá trên cùng một tập chỉ mục thời gian đồng nhất (4,900 giờ kiểm định độc lập).
4. **Kết quả nổi bật**: Mô hình **XGBoost** đạt sai số nhỏ nhất trên tập Validation (MAE = 0.3706 kW) và được lựa chọn làm mô hình đại diện. Trên tập Test độc lập, XGBoost đạt **MAE = 0.3242 kW**, **RMSE = 0.4666 kW**, $R^2 = 0.5698$.

---

## 1. Phương Pháp Luận & Quy Ước Bài Toán

### 1.1. Quy Ước Thời Gian & Tính Nhân Quả (Causal Integrity)
- Mỗi hàng dữ liệu được đánh chỉ mục bởi thời điểm bắt đầu khoảng mục tiêu $s$.
- Giá trị mục tiêu $y_s$ là công suất tác dụng trung bình trong khoảng 1 giờ $[s, s+1\text{h})$, được tổng hợp theo quy ước nhãn trái (`closed='left'`, `label='left'`).
- Thời điểm phát dự báo là $s$. Toàn bộ đặc trưng đầu vào $X_s$ chỉ được tính toán từ các quan sát diễn ra trước $s$ (tức các giờ $\le s-1$).
- Chuỗi thời gian được thiết lập trên chỉ mục liên tục hàng giờ (`pd.date_range(..., freq='1h')`) trước khi tạo bất kỳ biến trễ hay biến trượt nào, đảm bảo khoảng cách vật lý thực tế không bị co giãn khi gặp các khoảng thời gian gián đoạn đo đạc.

### 1.2. Phân Biệt Khái Niệm Công Suất (kW) và Điện Năng (kWh)
- **Công suất tác dụng ($P$, kW)**: Đại lượng đo tốc độ chuyển hóa điện năng tức thời. Trong bài toán này, giá trị đo đạc là trung bình số học của các phút trong giờ.
- **Điện năng tiêu thụ ($E$, kWh)**: Năng lượng tích lũy tiêu hao trong khoảng thời gian $t$:
  $$E = \int_{0}^{t} P(\tau) d\tau \approx P_{\text{tb}} \times t$$
  Vì khoảng thời gian dự báo ở đây cố định bằng đúng 1 giờ ($t = 1\text{ h}$), giá trị công suất trung bình $1.5\text{ kW}$ tương ứng với $1.5\text{ kWh}$ điện năng tiêu thụ trong khung giờ đó.

---

## 2. Kiểm Soát Chất Lượng Dữ Liệu & Tiền Xử Lý

### 2.1. Thống Kê Dữ Liệu Gốc (Data Quality Statistics)
- Tổng số bản ghi phút thô: **2,075,259 phút**.
- Số phút thiếu đo đạc công suất (`?` hoặc rỗng): **25,979 phút (1.25%)**.
- Số bản ghi trùng lặp timestamp: **0**.
- Mốc thời gian bắt đầu: `2006-12-16 17:24:00`.
- Mốc thời gian kết thúc: `2010-11-26 21:02:00`.

### 2.2. Chuẩn Hóa Chuỗi Giờ & Tiêu Chuẩn Nhãn Thực Tế
Sau khi tổng hợp sang tần suất 1 giờ liên tục:
- Tổng số khoảng 1 giờ liên tục: **34,589 giờ**.
- **Tiêu chuẩn nhãn thực tế hợp lệ**: Một giờ $s$ chỉ được coi là có nhãn ground truth hợp lệ ($y_s$) khi có **đủ đúng 60 phút đo đạc không bị thiếu**.
  - Số giờ hợp lệ đạt chuẩn: **34,085 giờ (98.54%)**.
  - Số giờ không hợp lệ (có $\le 59$ phút đo đạc): **504 giờ**.
- **Quy tắc bảo tồn dữ liệu**: Toàn bộ 504 giờ không đủ 60 phút được gán $y_s = \text{NaN}$ và bị loại khỏi việc tính điểm đánh giá; không tổng hợp nhãn giả mạo.
- **Xử lý chuỗi đầu vào lịch sử (Causal Imputation)**: Đối với chuỗi đầu vào dùng để tính đặc trưng trễ, các khoảng thiếu được điền tiến có chặn (bounded forward-fill tối đa 3 giờ liên tiếp `ffill(limit=3)`). Đối với các giá trị thiếu còn lại sau 3 giờ, sử dụng `SimpleImputer(strategy='median')` được huấn luyện **độc quyền trên tập Train của từng fold**, không làm rò rỉ dữ liệu sang tập Validation hay Test. Do đó, các biến trễ baseline phản ánh chuỗi lịch sử được điền tiến có giới hạn này.

---

## 3. Kỹ Thuật Tạo Đặc Trưng Nhân Quả (Feature Engineering)

Toàn bộ 20 đặc trưng được thiết kế tuân thủ nguyên tắc không rò rỉ dữ liệu:

1. **Đặc trưng độ trễ ngắn hạn và dài hạn (Lag Features - 6 biến)**:
   - Trễ 1, 2, 3 giờ: $x_{s-1}, x_{s-2}, x_{s-3}$.
   - Trễ 24, 48 giờ: $x_{s-24}, x_{s-48}$.
   - Trễ 168 giờ: $x_{s-168}$.
2. **Đặc trưng thống kê trượt (Rolling Statistics - 6 biến)**:
   - Được tính toán trên chuỗi đã dịch chuyển 1 giờ (`shift(1)`), đảm bảo giá trị của giờ hiện tại $s$ không tham gia vào cửa sổ trượt:
     - Trung bình trượt: `rolling_mean_3`, `rolling_mean_24`, `rolling_mean_168`.
     - Độ lệch chuẩn trượt: `rolling_std_3`, `rolling_std_24`, `rolling_std_168`.
3. **Đặc trưng lịch và tuần hoàn (Calendar & Cyclic - 8 biến)**:
   - Biến số: `hour` ($0..23$), `dayofweek` ($0..6$), `month` ($1..12$), `is_weekend` ($0/1$).
   - Biến tuần hoàn hình sin/cosin để giữ tính liên tục tại điểm chuyển giao 23:59 sang 00:00:
     $$\sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \sin\left(\frac{2\pi \cdot \text{dayofweek}}{7}\right), \quad \cos\left(\frac{2\pi \cdot \text{dayofweek}}{7}\right)$$

*Giai đoạn khởi động (Warmup)*: 168 giờ đầu tiên của tập dữ liệu được loại trừ hoàn toàn khỏi mọi mô hình để đảm bảo các biến trễ $s-168$ luôn có đủ dữ liệu lịch sử xác thực.

---

## 4. Thiết Kế Phân Chia Dữ Liệu & Kiểm Định Mở Rộng

### 4.1. Phân Chia Thời Gian (Chronological Split Boundaries)
Các ranh giới được ấn định theo mốc thời gian lịch trước khi lọc các nhãn không hợp lệ:

| Phân Vùng | Tỷ Lệ | Mốc Bắt Đầu | Mốc Kết Thúc | Tổng Giờ Lịch | Giờ Đủ 60 Phút Hợp Lệ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Warmup** | - | 2006-12-16 17:00:00 | 2006-12-23 16:00:00 | 168 | - |
| **Train Set** | 70% | 2006-12-23 17:00:00 | 2009-09-22 14:00:00 | 24,094 | 23,906 |
| **Validation Set** | 15% | 2009-09-22 15:00:00 | 2010-04-25 17:00:00 | 5,163 | 5,065 |
| **First 85% (Refit)** | 85% | 2006-12-23 17:00:00 | 2010-04-25 17:00:00 | 29,257 | 28,971 |
| **Test Set** | 15% | 2010-04-25 18:00:00 | 2010-11-26 21:00:00 | 5,164 | 4,948 |

### 4.2. Ranh Giới 3 Fold Expanding Cross-Validation Trong Tập Train (70%)
- **Fold 1**: Train [`2006-12-23 17:00:00` $\rightarrow$ `2007-08-31 15:00:00`] (5,940 mẫu) | Val [`2007-08-31 16:00:00` $\rightarrow$ `2008-05-08 14:00:00`] (6,014 mẫu).
- **Fold 2**: Train [`2006-12-23 17:00:00` $\rightarrow$ `2008-05-08 14:00:00`] (11,954 mẫu) | Val [`2008-05-08 15:00:00` $\rightarrow$ `2009-01-14 13:00:00`] (6,010 mẫu).
- **Fold 3**: Train [`2006-12-23 17:00:00` $\rightarrow$ `2009-01-14 13:00:00`] (17,964 mẫu) | Val [`2009-01-14 14:00:00` $\rightarrow$ `2009-09-22 14:00:00`] (5,942 mẫu).

### 4.3. Mặt Nạ Đánh Giá Chung (Common Evaluation Mask)
Để so sánh công bằng giữa các mô hình học máy và mô hình chuẩn cơ sở:
- Một mốc thời gian $s$ nằm trong tập đánh giá chung khi và chỉ khi: (1) Nhãn $y_s$ có đủ 60 phút hợp lệ; (2) Tồn tại đầy đủ $x_{s-1}$ (cho Persistence) và $x_{s-24}$ (cho Seasonal Naive).
- **Trên tập Validation**:
  - Số giờ lịch: 5,163 giờ.
  - Số giờ nhãn không hợp lệ: 98 giờ.
  - Số giờ nhãn hợp lệ: 5,065 giờ.
  - Số giờ bị loại do thiếu baseline: 48 giờ.
  - **Mặt nạ chung (Common Mask)**: **5,017 giờ** (Đạt tỷ lệ bao phủ 97.17% giờ lịch và 99.05% giờ nhãn hợp lệ).
- **Trên tập Test**:
  - Số giờ lịch: 5,164 giờ.
  - Số giờ nhãn không hợp lệ: 216 giờ.
  - Số giờ nhãn hợp lệ: 4,948 giờ.
  - Số giờ bị loại do thiếu baseline: 48 giờ.
  - **Mặt nạ chung (Common Mask)**: **4,900 giờ** (Đạt tỷ lệ bao phủ 94.89% giờ lịch và 99.03% giờ nhãn hợp lệ).

---

## 5. Kết Quả Huấn Luyện & Thẩm Định Mô Hình

### 5.1. Kết Quả Kiểm Định Chéo Mở Rộng (Train Expanding CV)
Bảng kết quả trung bình và độ lệch chuẩn qua 3 fold mở rộng trên tập Train (trích xuất trực tiếp từ `cv_metrics.csv`):

| Mô Hình | Cấu Hình Tối Ưu Lựa Chọn | Mean CV MAE (kW) | Std CV MAE | Mean CV RMSE (kW) | Mean CV $R^2$ |
|:---|:---|:---:|:---:|:---:|:---:|
| **Linear Regression** | `fit_intercept: True` | 0.4105 | 0.0514 | 0.5903 | 0.5588 |
| **Ridge** | `alpha: 10.0` | 0.4102 | 0.0518 | 0.5890 | 0.5611 |
| **Random Forest** | `n_estimators: 80, max_depth: 8, min_samples_split: 10` | 0.3853 | 0.0507 | 0.5617 | 0.6016 |
| **Gradient Boosting** | `n_estimators: 100, learning_rate: 0.05, max_depth: 4` | 0.3820 | 0.0516 | 0.5553 | 0.6106 |
| **XGBoost** | `n_estimators: 100, learning_rate: 0.05, max_depth: 5, subsample: 0.8` | 0.3760 | 0.0505 | 0.5500 | 0.6179 |

### 5.2. Đánh Giá Trên Tập Validation & Lựa Chọn Mô Hình Chiến Thắng
Mô hình sau khi được cố định siêu tham số tốt nhất từ CV được huấn luyện trên toàn bộ tập Train (70%) và đánh giá trên tập Validation (15%):

| Mô Hình | Validation MAE (kW) | Validation RMSE (kW) | Validation $R^2$ | % Cải Thiện vs Persistence | % Cải Thiện vs Seasonal Naive | Thời Gian Fit (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Persistence | 0.4636 | 0.6917 | 0.3941 | -0.00% | +30.17% | 0.00 |
| Seasonal Naive | 0.6639 | 0.9549 | -0.1547 | -43.21% | +0.00% | 0.00 |
| Linear Regression | 0.4152 | 0.5854 | 0.5662 | +10.44% | +37.46% | 0.08 |
| Ridge | 0.4152 | 0.5853 | 0.5662 | +10.44% | +37.46% | 0.07 |
| Random Forest | 0.3740 | 0.5456 | 0.6231 | +19.31% | +43.66% | 2.91 |
| Gradient Boosting | 0.3756 | 0.5415 | 0.6288 | +18.97% | +43.42% | 10.55 |
| XGBoost | 0.3706 | 0.5368 | 0.6351 | +20.04% | +44.17% | 0.39 |

> **Quyết định thẩm định**: **XGBoost** đạt MAE nhỏ nhất (0.3706 kW) trên tập Validation, được chính thức lựa chọn làm **Validation Winner** trước khi bước vào đánh giá trên tập Test.

---

## 6. Kết Quả Đánh Giá Trên Tập Test (Held-Out Test Results)

Mô hình được tái huấn luyện trên 85% dữ liệu đầu (Train + Val, gồm 28,971 giờ hợp lệ) với cấu hình siêu tham số cố định và đánh giá trên tập Test (4,900 giờ kiểm định). Lưu ý rằng trong quá trình phát triển dự án, việc thực thi lại quy trình đã được thực hiện phục vụ mục đích kiểm định kỹ thuật mã nguồn mà không thay đổi bất kỳ siêu tham số nào theo tập test.

| STT | Mô Hình | MAE (kW) | RMSE (kW) | $R^2$ Score | Cải Thiện vs Persistence | Cải Thiện vs Seasonal Naive | Thời Gian Fit (s) | Số Mẫu Test |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Persistence | 0.3892 | 0.5885 | 0.3155 | +0.00% | +23.68% | 0.00 | 4,900 |
| 2 | Seasonal Naive | 0.5100 | 0.7601 | -0.1418 | -31.03% | +0.00% | 0.00 | 4,900 |
| 3 | Linear Regression | 0.3567 | 0.5022 | 0.5014 | +8.34% | +30.04% | 0.07 | 4,900 |
| 4 | Ridge | 0.3567 | 0.5022 | 0.5015 | +8.34% | +30.04% | 0.08 | 4,900 |
| 5 | Random Forest | 0.3311 | 0.4747 | 0.5547 | +14.93% | +35.08% | 3.72 | 4,900 |
| 6 | Gradient Boosting | 0.3279 | 0.4707 | 0.5622 | +15.74% | +35.70% | 13.21 | 4,900 |
| 7 | **XGBoost** | **0.3242** | **0.4666** | **0.5698** | **+16.70%** | **+36.42%** | **0.31** | 4,900 |

### Nhận Xét Kết Quả Chính:
1. **Mô hình học máy dạng cây đạt kết quả vượt trội**: XGBoost đạt MAE 0.3242 kW, cải thiện +16.70% so với mô hình Persistence và +36.42% so với Seasonal Naive.
2. **Persistence tốt hơn Seasonal Naive**: Mô hình Persistence (lấy giờ liền trước $s-1$, MAE 0.3892 kW) tốt hơn Seasonal Naive (lấy cùng giờ ngày hôm trước $s-24$, MAE 0.5100 kW) +23.68%. Điều này cho thấy tính tương quan tự hồi quy cực ngắn hạn (1 giờ trước) mạnh hơn so với tính chu kỳ 24 giờ của một hộ gia đình đơn lẻ.

---

## 7. Nghiên Cứu Cắt Bỏ Thành Phần (Ablation Study)

Thử nghiệm Ablation được tiến hành trên mô hình **XGBoost** trên tập Validation và tập Test với thứ tự các cột đặc trưng đồng nhất hoàn toàn:

| Nhóm Cấu Hình | Số Biến | Val MAE (kW) | Val RMSE (kW) | Val $R^2$ | Test MAE (kW) | Test RMSE (kW) | Test $R^2$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1. Calendar Only (Chi Lịch) | 8 | 0.5046 | 0.6787 | 0.4167 | 0.4261 | 0.5848 | 0.3241 |
| 2. Lag & Rolling Only (Chi Chuoi) | 12 | 0.3957 | 0.5704 | 0.5881 | 0.3447 | 0.4877 | 0.5299 |
| 3. Full minus Rolling (Lich + Lags) | 14 | 0.3724 | 0.5385 | 0.6328 | 0.3288 | 0.4698 | 0.5638 |
| **4. Full Features (Toan bo 20 dac trung)** | 20 | **0.3706** | 0.5368 | 0.6351 | **0.3242** | 0.4666 | 0.5698 |

### Đóng Góp Của Từng Nhóm Đặc Trưng:
- **Vai trò của nhóm trễ và chuỗi thời gian**: Khi chỉ sử dụng các biến lịch (cấu hình `Calendar Only`), sai số MAE trên Validation tăng lên 0.5046 kW (và trên Test là 0.4261 kW), cho thấy nhóm thông tin trễ và trượt chuỗi thời gian đóng vai trò quyết định.
- **Sự kết hợp đầy đủ**: Trên tập Test, cấu hình đầy đủ `Full Features` đạt MAE 0.3242 kW, cho thấy việc kết hợp cả đặc trưng trễ, đặc trưng trượt và đặc trưng lịch tuần hoàn mang lại độ chính xác tổng thể tốt nhất.

---

## 8. Phân Tích Sai Số Phụ Tải Cao (High-Load Error Analysis)

Ngưỡng phụ tải cao được xác định tại **phân vị 90 (P90) của tập Train**: **2.3800 kW**.

Đánh giá độ chính xác của các mô hình trên tập Test được phân tách thành hai miền phụ tải:

| Mô Hình | Giờ Cao Điểm ($\\ge 2.3800\\text{ kW}$) MAE | Giờ Cao Điểm RMSE | Giờ Bình Thường MAE | Giờ Bình Thường RMSE | Tổng Thể MAE |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Persistence** | 1.0886 | 1.3140 | 0.3554 | 0.5288 | 0.3892 |
| **Seasonal Naive** | 1.3850 | 1.6555 | 0.4676 | 0.6878 | 0.5100 |
| **Linear Regression** | 1.1190 | 1.2850 | 0.3199 | 0.4297 | 0.3567 |
| **Ridge** | 1.1193 | 1.2852 | 0.3199 | 0.4296 | 0.3567 |
| **Random Forest** | 1.0093 | 1.1861 | 0.2983 | 0.4101 | 0.3311 |
| **Gradient Boosting** | 1.0105 | 1.1833 | 0.2949 | 0.4056 | 0.3279 |
| **XGBoost** | 1.0028 | 1.1757 | 0.2914 | 0.4017 | 0.3242 |

### Thảo Luận:
- Trong các khung giờ bình thường, XGBoost đạt MAE 0.2914 kW.
- Khi phụ tải vượt ngưỡng P90, sai số MAE tăng lên 1.0028 kW.
- *Giả thuyết phân tích*: Sự gia tăng sai số ở miền tải cao phản ánh tính bất định lớn hơn trong các giai đoạn hoạt động ngẫu nhiên của các thiết bị công suất cao.

---

## 9. Cơ Sở Lý Thuyết Của Các Mô Hình Nghiên Cứu

1. **Persistence (Naive Lag-1)**: Giả định $\hat{y}_s = x_{s-1}$. Đây là mô hình cơ sở trong dự báo chuỗi thời gian, phản ánh tính trễ ngắn hạn.
2. **Seasonal Naive (Lag-24)**: Giả định $\hat{y}_s = x_{s-24}$. Phản ánh tính lặp lại chu kỳ 24 giờ.
3. **Linear Regression (OLS)**: Ước lượng hồi quy tuyến tính cổ điển $\hat{y}_s = w_0 + \sum w_i X_{s,i}$ tối thiểu hóa tổng bình phương phần dư (MSE).
4. **Ridge Regression ($L_2$ Regularization)**: Bổ sung ràng buộc độ lớn hệ số $\alpha \sum w_i^2$ vào hàm mất mát, giúp kiểm soát hiện tượng đa cộng tuyến giữa các biến trễ liên tiếp.
5. **Random Forest Regressor**: Mô hình kết hợp biểu quyết (Bagging) từ nhiều cây quyết định độc lập trên các tập thuộc tính ngẫu nhiên, giúp kiểm soát phương sai của mô hình.
6. **Gradient Boosting Regressor**: Thuật toán tăng cường xây dựng tuần tự các cây quyết định, trong đó mỗi cây mới được tối ưu để khớp với phần dư từ các cây trước đó.
7. **XGBoost (Extreme Gradient Boosting)**: Cải tiến Gradient Boosting với xấp xỉ chuỗi Taylor bậc hai của hàm mục tiêu, kết hợp điều chuẩn độ phức tạp của cây và tối ưu hóa tính toán trên phần cứng.

---

## 10. Bộ Câu Hỏi & Trả Lời Bảo Vệ Đồ Án (Defense Q&A)

### Q1: Tại sao phải sử dụng mô hình trễ 1 giờ ($[s, s+1\text{h})$ dự báo tại $s$ bằng dữ liệu $< s$) thay vì dùng các thông số cùng giờ?
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
