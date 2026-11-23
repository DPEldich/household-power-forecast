Em chào thầy cô. Em xin trình bày chương trình dự báo công suất tiêu thụ điện hộ gia đình trong một giờ tiếp theo. Mục tiêu của đồ án là sử dụng dữ liệu tiêu thụ điện trong quá khứ để ước lượng mức sử dụng điện sắp tới, đồng thời xây dựng giao diện để quan sát và kiểm tra kết quả dự báo.

Đầu ra của chương trình là công suất trung bình, đơn vị kilowatt. Ví dụ, tại thời điểm 10 giờ, chương trình dùng dữ liệu trước 10 giờ để dự báo công suất trung bình từ 10 đến 11 giờ. Với khoảng thời gian một giờ, công suất trung bình 1,5 kilowatt tương ứng với điện năng tiêu thụ 1,5 kilowatt giờ.

Dữ liệu được lấy từ kho UCI, ghi nhận hoạt động sử dụng điện của một hộ gia đình tại Pháp trong giai đoạn 2006 đến 2010. Tập dữ liệu có hơn hai triệu bản ghi theo phút, được tổng hợp thành dữ liệu theo giờ. Khi đánh giá, em chỉ sử dụng những giờ có đủ 60 phút đo hợp lệ. Những giờ thiếu dữ liệu không được tự điền thành giá trị thực tế để tính điểm.

Từ dữ liệu này, em tạo các đặc trưng như mức tiêu thụ của những giờ gần nhất, cùng giờ ngày hôm trước và tuần trước, trung bình trượt, độ biến động và thông tin lịch. Nguyên tắc quan trọng là chỉ sử dụng thông tin đã có tại thời điểm dự báo, tránh để dữ liệu tương lai lọt vào đầu vào của mô hình.

Em chia dữ liệu theo thứ tự thời gian: 70% đầu để huấn luyện, 15% tiếp theo để lựa chọn mô hình và 15% cuối để kiểm tra. Trong phần huấn luyện, em sử dụng kiểm định chéo mở rộng theo thời gian. Cách chia này mô phỏng việc học từ quá khứ để dự báo cho giai đoạn phía sau.

Chương trình so sánh bảy mô hình, gồm hai phương pháp cơ sở, hồi quy tuyến tính, Ridge, Random Forest, Gradient Boosting và XGBoost. Hai phương pháp cơ sở lần lượt lấy mức tiêu thụ của giờ trước và cùng giờ hôm trước làm dự báo. XGBoost được chọn vì có sai số tuyệt đối trung bình thấp nhất trên tập lựa chọn mô hình. Sau đó, mô hình được huấn luyện lại trên 85% dữ liệu đầu và đánh giá trên phần cuối.

Trên cùng 4.900 giờ kiểm tra, XGBoost đạt MAE khoảng 0,324 kilowatt và RMSE khoảng 0,467 kilowatt. MAE nghĩa là độ lệch tuyệt đối giữa dự báo và thực tế trung bình khoảng 0,324 kilowatt. So với phương pháp lấy giá trị giờ trước, MAE giảm khoảng 16,7%; so với lấy cùng giờ hôm trước, mức giảm khoảng 36,4%.

Sau đây em xin minh họa trên giao diện. Ở bảng kết quả, thầy cô có thể so sánh sai số giữa các mô hình. Khi chuyển sang phần so sánh chuỗi lịch sử và dự báo, em có thể chọn khoảng thời gian để quan sát đường dự báo bám theo thực tế và nhận ra những thời điểm sai lệch lớn.

Trong phần mô phỏng, em chọn một giờ mục tiêu và mô hình XGBoost, sau đó bấm thực thi suy luận trực tiếp. Giao diện Streamlit gửi yêu cầu đến FastAPI. Backend tạo đặc trưng từ lịch sử, chạy mô hình đã lưu và trả kết quả. Em có thể bật hiển thị giá trị thực tế để đối chiếu sai số. Bước này minh họa việc gọi mô hình để tính dự báo, bên cạnh việc xem bảng kết quả đã lưu.

Chương trình còn có phần mở rộng dự báo khoảng P10 đến P90, giúp thể hiện mức độ bất định. Tuy nhiên, khoảng này vẫn bỏ sót nhiều giờ có phụ tải cao. Thử nghiệm bổ sung đặc trưng cũng chưa đáp ứng điều kiện cải thiện sai số ở nhóm phụ tải cao trên tập xác nhận, nên em giữ mô hình XGBoost ban đầu cho dự báo điểm.

Giới hạn hiện tại là chương trình đang chạy lại dữ liệu lịch sử của một hộ gia đình, chưa kết nối công tơ điện thời gian thực. Mỗi lần chương trình dự báo một giờ tiếp theo từ lịch sử sẵn có; các đỉnh tiêu thụ đột ngột vẫn khó dự báo. Hướng phát triển là bổ sung dữ liệu từ nhiều hộ và cải thiện dự báo trong những giờ cao điểm. Em xin kết thúc phần trình bày và cảm ơn thầy cô.
