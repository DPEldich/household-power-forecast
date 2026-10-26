# ML-03 — Nghiệm thu độc lập của Codex

Ngày: 2026-10-26, Asia/Saigon. Trạng thái: **đã nghiệm thu phần triển khai và thực nghiệm**.

Coding dùng skill antigravity-task-runner theo yêu cầu người dùng: `gemini-3.8-flash-high`, hội thoại `38f8b4cf-c23d-4afa-96a7-f49782408c6a`, chạy tuần tự và qua hai vòng sửa lỗi. Codex đọc diff, kiểm tra source/tests/docs và tự thực hiện các kiểm tra dưới đây. Lần gọi CLI đầu chạm giới hạn chờ đầu ra 20 phút; tiếp tục cùng hội thoại, không đổi mô hình.

## Bằng chứng nghiệm thu

- Trước thay đổi: 48 test đạt; baseline Git `5276c29`, working tree sạch.
- Bản cuối: `.venv/Scripts/python.exe -m pytest -q` — **80 passed, 3 warnings, 37.29s**. Warnings sẵn có: Starlette/httpx deprecation và hai cảnh báo vẽ notebook không có GUI.
- `.venv/Scripts/python.exe -m pip check` — không có xung đột phụ thuộc.
- `.venv/Scripts/python.exe scripts/export_serving_bundle.py` — release **uci-hourly-9c37e566d5f6e848**, kiểm tra toàn bộ 4.900 dòng của bảy mô hình gốc và bốn mô hình điểm/phân vị mới.
- SHA256 của **16 tệp mô hình/metrics/predictions/metadata nền** không đổi so với snapshot trước coding; snapshot tại `logs/ml03-baseline-hashes.json` (local, ignored).
- Tự tính lại từ CSV: số mẫu, coverage, độ rộng và từng pinball loss của cả sáu nhóm validation/test; MAE tổng thể/tải cao của baseline và ứng viên trên validation; ngưỡng tải cao của ba fold từ dữ liệu train tương ứng. Tất cả khớp.
- Tự tái hiện trường hợp đảo thứ tự đặc trưng ở cả manifest và metadata, cập nhật checksum hợp lệ: khởi động bị từ chối vì không khớp tên cột của pipeline đã huấn luyện.
- Kiểm tra tính nhân quả và sự bằng nhau của 20 đặc trưng nền trong bộ 26 đặc trưng; đã xem hình `artifacts/experimental/ml03/figures/one_week_interval_forecast.png`.
- Test bao phủ API mới/cũ, dữ liệu lịch sử tự cung cấp, checksum/path/feature contract, bundle legacy, giao diện qua HTTP thật, phản hồi sai phiên bản/sai dự báo và bảng số liệu đầy đủ trong ba tài liệu.
- Kiểm tra whitespace của diff đạt khi cho phép CRLF của Windows và ngắt dòng Markdown có chủ ý.

## Kết luận thực nghiệm

Ứng viên mở rộng không được thay thế mô hình nền: validation MAE tổng thể giảm từ 0.370641 xuống 0.369626 kW, nhưng MAE tải cao tăng từ 0.835970 lên 0.840934 kW, vi phạm điều kiện xác nhận đã định trước. XGBoost nền tiếp tục dự báo điểm, test MAE 0.324208 kW.

P10/P50/P90 đã được huấn luyện, đánh giá, lưu và phục vụ qua API/giao diện. Trên benchmark test lịch sử dùng lại, khoảng P10–P90 phủ 90.49% tổng thể nhưng chỉ 38.05% giờ tải cao; đây là kết quả thử nghiệm, chưa có bảo đảm hiệu chuẩn. P50 là trung vị, khác mục tiêu squared-error của dự báo điểm.

Các lỗi phát hiện ở vòng review đã được sửa trước closeout. Không còn blocker kỹ thuật trong phạm vi ML-03. Tính hợp lệ ngoài một hộ gia đình, dữ liệu mới, dự báo 24 giờ và triển khai cloud chưa được đánh giá trong đợt này. Chưa commit hoặc push. Báo cáo chi tiết: [ML03_RESULTS.md](ML03_RESULTS.md).
