"""Giao diện Streamlit tương tác cho Bài tập lớn môn học Machine Learning:
Dự báo công suất tiêu thụ điện hộ gia đình theo giờ tiếp theo.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Đảm bảo src luôn có trong sys.path để ứng dụng hoàn toàn độc lập khi chạy ở bất kỳ đâu
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from household_power.api_client import ApiError, request_json

st.set_page_config(
    page_title="Dự Báo Tiêu Thụ Điện Hộ Gia Đình (ML-01)",
    page_icon="⚡",
    layout="wide",
)

import os

def setting(name: str, default: str = "") -> str:
    if name in os.environ:
        return os.environ[name]
    try:
        return str(st.secrets.get(name, default))
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        return default


API_BASE_URL = setting("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = setting("API_KEY")


@st.cache_data(ttl=300, show_spinner=False)
def load_dashboard(base_url: str, api_key: str):
    return request_json(base_url, "/v1/dashboard", api_key)


@st.cache_data(ttl=300, show_spinner=False)
def load_history(base_url: str, api_key: str, target: str, hours: int, version: str):
    payload = request_json(base_url, "/v1/history", api_key,
                           params={"target_timestamp": target, "hours": hours})
    if payload["model_version"] != version:
        raise ApiError("Backend đã đổi phiên bản mô hình. Hãy làm mới dữ liệu API.")
    frame = pd.DataFrame(payload["history"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.set_index("timestamp")


# ==================== MAIN UI ====================
st.title("Dự Báo Tiêu Thụ Điện Hộ Gia Đình Theo Giờ Tiếp Theo (Next-Hour Forecasting)")
st.caption(
    "Bài tập lớn kết thúc học phần Machine Learning | "
    "Dữ liệu nguồn: UCI Machine Learning Repository (Dataset 235: 2006 – 2010)"
)
st.caption(
    "Frontend Streamlit · Backend FastAPI · Dự báo từ mô hình đã huấn luyện"
)

st.info(
    "Lưu ý: Ứng dụng hoạt động ở chế độ chạy lại dữ liệu kiểm định lịch sử (Historical Test Replay), "
    "sử dụng dữ liệu đo đạc thực tế của 01 hộ gia đình tại Pháp trong giai đoạn 2006 – 2010."
)

if st.sidebar.button("Làm mới dữ liệu API", key="refresh_api"):
    st.cache_data.clear()
    st.rerun()

try:
    with st.spinner("Đang kết nối FastAPI; server miễn phí có thể cần khoảng một phút để thức dậy…"):
        snapshot = load_dashboard(API_BASE_URL, API_KEY)
    if snapshot.get("schema_version") != 1:
        raise ApiError("Phiên bản API chưa tương thích với giao diện này.")
    metadata = snapshot["metadata"]
    model_version = snapshot["model_version"]
    metrics_dict = {name: pd.DataFrame(rows) for name, rows in snapshot["metrics"].items()}
    test_preds = pd.DataFrame(snapshot["predictions"])
    test_preds["timestamp"] = pd.to_datetime(test_preds["timestamp"])
    st.sidebar.caption(f"Phiên bản mô hình: {model_version}")
except (ApiError, KeyError, ValueError) as exc:
    st.error(str(exc))
    st.info("Kiểm tra địa chỉ API_BASE_URL. Khi chạy local, mở FastAPI ở cổng 8000 trước. Khi backend miễn phí đang ngủ, thử lại sau khoảng một phút.")
    st.stop()

if "validation_selected_winner" not in metadata:
    st.error("Lỗi cấu trúc metadata: Không tìm thấy trường 'validation_selected_winner'. Vui lòng chạy lại pipeline.")
    st.stop()

winner_name = metadata["validation_selected_winner"]

if winner_name not in metrics_dict["test"]["model_name"].values:
    st.error("Metadata và bảng kết quả không cùng phiên bản. Hãy làm mới dữ liệu API.")
    st.stop()
test_df = metrics_dict.get("test")
val_df = metrics_dict.get("val")

# Top summary KPIs
col1, col2, col3, col4, col5 = st.columns(5)
winner_test_row = test_df[test_df["model_name"] == winner_name].iloc[0] if test_df is not None else None
persistence_row = test_df[test_df["model_name"] == "Persistence"].iloc[0] if test_df is not None else None
s_naive_row = test_df[test_df["model_name"] == "Seasonal Naive"].iloc[0] if test_df is not None else None

with col1:
    st.metric("Mô hình tốt nhất (Validation)", winner_name)
with col2:
    if winner_test_row is not None:
        st.metric("MAE trên Test (kW)", f"{winner_test_row['mae']:.4f}")
with col3:
    if winner_test_row is not None:
        st.metric("RMSE trên Test (kW)", f"{winner_test_row['rmse']:.4f}")
with col4:
    if winner_test_row is not None:
        st.metric("Cải thiện vs Persistence", f"{winner_test_row['mae_improvement_vs_persistence_pct']:+.2f}%")
with col5:
    if winner_test_row is not None:
        st.metric("Cải thiện vs Seasonal Naive", f"{winner_test_row['mae_improvement_vs_seasonal_naive_pct']:+.2f}%")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Bảng số liệu Held-Out Test",
    "2. So sánh chuỗi lịch sử & Dự báo",
    "3. Mô phỏng & Suy luận từ mô hình đã lưu",
    "4. Phân tích sai số theo giờ",
    "5. Nghiên cứu cắt bỏ đặc trưng (Ablation)",
    "6. Phương pháp luận & Giới hạn",
])

# ==================== TAB 1: METRICS ====================
with tab1:
    st.header("Kết quả Đánh giá Trên Tập Kiểm Định Độc Lập (Held-Out Test Set)")
    st.write(
        f"Mô hình được lựa chọn dựa trên **MAE nhỏ nhất trên tập Validation (15% theo thời gian)**: **{winner_name}**. "
        "Mô hình được tái huấn luyện trên 85% dữ liệu đầu (Train + Val) và chỉ đánh giá đúng một lần duy nhất trên "
        "15% dữ liệu cuối cùng (Test set) với cùng mặt nạ thời gian chung (Common Mask)."
    )

    if test_df is not None:
        display_cols = [
            "model_name", "mae", "rmse", "r2",
            "mae_improvement_vs_persistence_pct",
            "mae_improvement_vs_seasonal_naive_pct",
            "fit_time_seconds", "sample_count",
        ]
        rename_map = {
            "model_name": "Tên Mô Hình",
            "mae": "MAE (kW)",
            "rmse": "RMSE (kW)",
            "r2": "R² Score",
            "mae_improvement_vs_persistence_pct": "% Cải thiện vs Persistence",
            "mae_improvement_vs_seasonal_naive_pct": "% Cải thiện vs Seasonal Naive",
            "fit_time_seconds": "Thời gian Huấn luyện (s)",
            "sample_count": "Số mẫu đánh giá",
        }
        formatted_df = test_df[display_cols].rename(columns=rename_map).copy()
        for col in ["MAE (kW)", "RMSE (kW)", "R² Score"]:
            formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.4f}")
        for col in ["% Cải thiện vs Persistence", "% Cải thiện vs Seasonal Naive"]:
            formatted_df[col] = formatted_df[col].map(lambda x: f"{x:+.2f}%")
        formatted_df["Thời gian Huấn luyện (s)"] = formatted_df["Thời gian Huấn luyện (s)"].map(lambda x: f"{x:.2f}")

        st.dataframe(formatted_df, width="stretch")

        fig_comp = px.bar(
            test_df,
            x="model_name",
            y=["mae", "rmse"],
            barmode="group",
            labels={"value": "Sai số (kW)", "model_name": "Mô hình", "variable": "Chỉ số"},
            title="So sánh MAE và RMSE giữa 7 mô hình trên tập Test",
            color_discrete_map={"mae": "#1f77b4", "rmse": "#aec7e8"},
        )
        st.plotly_chart(fig_comp, width="stretch")

    if "high_load" in metrics_dict:
        st.subheader("Phân tích sai số theo mức phụ tải cao (High-Load Analysis)")
        hl_df = metrics_dict["high_load"]
        st.write(
            f"Ngưỡng phụ tải cao được tính cố định tại **phân vị 90 (P90) của tập Train**: **{metadata.get('high_load_threshold_train_p90_kW', 0):.4f} kW**.\n"
            "Bảng dưới đây so sánh sai số khi hộ gia đình sử dụng điện cao điểm (>= P90) so với các giờ phụ tải thông thường."
        )
        st.dataframe(hl_df, width="stretch")

# ==================== TAB 2: HISTORICAL ACTUAL VS FORECAST ====================
with tab2:
    st.header("So sánh Chuỗi Thời Gian Lịch Sử & Dự Báo (Test Set)")
    st.write("Quan sát đồ thị đường biểu diễn thực tế và các mô hình dự báo trong giai đoạn kiểm định.")

    all_models = [c for c in test_preds.columns if c not in ["timestamp", "actual"]]
    selected_models = st.multiselect(
        "Chọn các mô hình muốn so sánh trên đồ thị:",
        options=all_models,
        default=[winner_name, "Persistence", "Seasonal Naive"] if winner_name in all_models else all_models[:3],
    )

    total_pts = len(test_preds)
    window_days = st.slider("Số ngày hiển thị liên tiếp:", min_value=3, max_value=60, value=14, step=1)
    window_hours = window_days * 24

    max_start = max(0, total_pts - window_hours)
    start_pos = st.slider("Điểm bắt đầu quan sát (giờ):", min_value=0, max_value=max_start, value=min(100, max_start), step=24)

    start_ts = test_preds["timestamp"].iloc[start_pos]
    end_ts = start_ts + pd.Timedelta(days=window_days)
    continuous_ts_range = pd.date_range(start_ts, end_ts - pd.Timedelta(hours=1), freq="1h")
    sub_pred = test_preds.set_index("timestamp").reindex(continuous_ts_range).reset_index().rename(columns={"index": "timestamp"})

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=sub_pred["timestamp"],
        y=sub_pred["actual"],
        mode="lines",
        name="Thực tế (Ground Truth)",
        line=dict(color="#d62728", width=2.4),
    ))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    for i, m_name in enumerate(selected_models):
        c = colors[i % len(colors)]
        dash_style = "dash" if "Naive" in m_name or "Persistence" in m_name else "solid"
        fig_ts.add_trace(go.Scatter(
            x=sub_pred["timestamp"],
            y=sub_pred[m_name],
            mode="lines",
            name=f"Dự báo: {m_name}",
            line=dict(color=c, width=1.6, dash=dash_style),
        ))

    fig_ts.update_layout(
        title=f"So sánh Dự Báo vs Thực Tế ({window_days} ngày trong tập Test)",
        xaxis_title="Thời gian (Giờ bắt đầu khoảng [s, s+1h))",
        yaxis_title="Công suất tác dụng trung bình (kW)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ts, width="stretch")

# ==================== TAB 3: REPLAY & LIVE INFERENCE ====================
with tab3:
    st.header("Mô Phỏng Từng Giờ & Suy Luận Trực Tiếp Từ Mô Hình Đã Lưu")
    st.info(
        "Nguyên tắc nhân quả (Causal Integrity): Tại thời điểm s, mô hình chỉ được biết thông tin tiêu thụ điện "
        "của các giờ trước đó (< s) để dự báo cho khoảng thời gian [s, s+1h). Không có rò rỉ dữ liệu cùng giờ hoặc tương lai."
    )

    max_idx = max(0, len(test_preds) - 1)
    default_row = min(240, max_idx)
    row_idx = st.number_input(
        f"Chọn vị trí giờ mục tiêu s trong tập Test (0 đến {max_idx}):",
        min_value=0,
        max_value=max_idx,
        value=default_row,
        step=1,
    )

    selected_row = test_preds.iloc[row_idx]
    target_time = selected_row["timestamp"]
    actual_val = float(selected_row["actual"])

    st.markdown(f"### Giờ mục tiêu phát dự báo s: `{target_time}`")

    # Chronological history before s
    st.subheader("1. Lịch sử tiêu thụ phụ tải trước thời điểm s")
    history_hours = st.slider("Số giờ lịch sử hiển thị:", min_value=12, max_value=72, value=24, step=6)

    try:
        hourly_df_all = load_history(API_BASE_URL, API_KEY, target_time.isoformat(), history_hours, model_version)
        # Pull continuous chronological history strictly before target_time
        hist_end = target_time
        hist_start = target_time - pd.Timedelta(hours=history_hours)
        hist_slice = hourly_df_all.loc[(hourly_df_all.index >= hist_start) & (hourly_df_all.index < hist_end)]

        if len(hist_slice) > 0:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=hist_slice.index,
                y=hist_slice["raw_active_power"],
                mode="lines+markers",
                name="Lịch sử quá khứ (< s)",
                line=dict(color="#1f77b4", width=2),
            ))
            fig_hist.add_vline(x=target_time, line_width=2, line_dash="dash", line_color="red")
            fig_hist.add_annotation(
                x=target_time, y=hist_slice["raw_active_power"].dropna().max() if len(hist_slice["raw_active_power"].dropna()) > 0 else 1.0,
                text="Thời điểm phát dự báo s", showarrow=True, arrowhead=1
            )
            fig_hist.update_layout(
                title=f"Chuỗi phụ tải lịch sử {len(hist_slice)} giờ liên tục trước thời điểm s",
                xaxis_title="Thời gian",
                yaxis_title="Công suất (kW)",
                height=320,
            )
            st.plotly_chart(fig_hist, width="stretch")
    except ApiError as exc:
        st.warning(str(exc))

    st.subheader("2. Giá trị dự báo của các mô hình cho khoảng [s, s+1h)")
    pred_display = {}
    for col in all_models:
        pred_display[col] = float(selected_row[col])

    df_p_disp = pd.DataFrame([pred_display], index=["Dự báo (kW)"]).T
    st.table(df_p_disp)

    reveal = st.checkbox("Tiết lộ giá trị thực tế đo đạc tại giờ s (Reveal Ground Truth)")
    if reveal:
        st.write(f"**Giá trị thực tế đo đạc y_s:** `{actual_val:.4f} kW`")
        diff_df = pd.DataFrame({
            "Mô hình": list(pred_display.keys()),
            "Dự báo (kW)": [f"{v:.4f}" for v in pred_display.values()],
            "Thực tế (kW)": f"{actual_val:.4f}",
            "Sai số tuyệt đối |y - ŷ| (kW)": [f"{abs(actual_val - v):.4f}" for v in pred_display.values()],
        })
        st.dataframe(diff_df, width="stretch")

    st.markdown("---")
    st.subheader("3. Dự báo trực tiếp qua FastAPI")
    st.write(
        "Streamlit gửi giờ mục tiêu và tên mô hình tới backend FastAPI. Backend tạo đặc trưng từ lịch sử, "
        "chạy mô hình đã lưu và trả dự báo kèm phiên bản để đối chiếu với bảng kiểm định."
    )

    test_model_choice = st.selectbox("Chọn mô hình để chạy suy luận trực tiếp:", options=all_models, key="predict_model")
    if st.button("Thực thi suy luận trực tiếp (Live Predict)", key="live_predict"):
        try:
            result = request_json(API_BASE_URL, "/v1/predict", API_KEY, body={
                "model": test_model_choice, "target_timestamp": target_time.isoformat(),
                "expected_model_version": model_version,
            })
            live_pred = float(result["prediction_kw"])
            precomputed_pred = float(selected_row[test_model_choice])
            diff = abs(live_pred - precomputed_pred)
            st.write(f"Dự báo FastAPI: **{live_pred:.6f} kW** · Phiên bản: `{result['model_version']}`")
            st.write(f"Dự báo đã lưu: {precomputed_pred:.6f} kW · Độ lệch: {diff:.10f} kW")
            if result["model_version"] != model_version:
                st.error("Backend đã đổi phiên bản. Hãy làm mới dữ liệu API.")
            elif diff <= 1e-4:
                st.success("XÁC MINH THÀNH CÔNG: Dự báo FastAPI trùng khớp với bảng kiểm định.")
            else:
                st.error(f"PHÁT HIỆN SAI LỆCH: {diff:.6f} kW so với bảng dự báo lưu trữ.")
        except ApiError as exc:
            st.error(str(exc))

# ==================== TAB 4: MAE BY HOUR ====================
with tab4:
    st.header("Phân Tích Sai Số MAE Theo Các Giờ Trong Ngày (Diurnal Error)")
    st.write(
        "Mức độ biến động của phụ tải thay đổi giữa các khung giờ trong ngày. "
        "Biểu đồ dưới đây thể hiện sai số MAE phân bổ theo 24 giờ trong ngày được tính toán trực tiếp từ dữ liệu kiểm định."
    )

    hours = test_preds["timestamp"].dt.hour
    hourly_err_records = []
    for m in all_models:
        mae_h = np.abs(test_preds["actual"] - test_preds[m]).groupby(hours).mean()
        for h, val in mae_h.items():
            hourly_err_records.append({"hour": h, "model": m, "mae": val})

    df_h_err = pd.DataFrame(hourly_err_records)

    # Dynamic calculation of best and worst hours for winner model
    winner_h_err = df_h_err[df_h_err["model"] == winner_name].sort_values("mae")
    best_h = int(winner_h_err.iloc[0]["hour"])
    best_mae = float(winner_h_err.iloc[0]["mae"])
    worst_h = int(winner_h_err.iloc[-1]["hour"])
    worst_mae = float(winner_h_err.iloc[-1]["mae"])

    fig_h = px.line(
        df_h_err,
        x="hour",
        y="mae",
        color="model",
        markers=True,
        labels={"hour": "Giờ trong ngày (0 – 23)", "mae": "MAE (kW)", "model": "Mô hình"},
        title="Sai số MAE trung bình theo từng giờ trong ngày trên tập Test",
    )
    fig_h.update_xaxes(tickmode="linear", tick0=0, dtick=1)
    st.plotly_chart(fig_h, width="stretch")

    st.markdown(
        f"""
        **Thống kê quan sát từ dữ liệu thực tế:**
        - Với mô hình `{winner_name}`, khung giờ có sai số MAE thấp nhất trong tập kiểm định là **giờ {best_h:02d}:00** ({best_mae:.4f} kW).
        - Khung giờ có sai số MAE cao nhất là **giờ {worst_h:02d}:00** ({worst_mae:.4f} kW).
        - *Giả thuyết phân tích*: Sự chênh lệch sai số giữa các khung giờ có thể phản ánh mức độ bất định trong hành vi sinh hoạt của hộ gia đình (các khung giờ nghỉ ngơi phụ tải ổn định hơn so với các khung giờ sinh hoạt cao điểm).
        """
    )

# ==================== TAB 5: ABLATION STUDY ====================
with tab5:
    st.header("Nghiên Cứu Cắt Bỏ Đặc Trưng (Ablation Study)")
    st.write(
        f"Ablation study được thực hiện trên **tập Validation** với mô hình **{winner_name}** "
        "để định lượng sự thay đổi sai số khi sử dụng các nhóm đặc trưng khác nhau."
    )

    if "ablation" in metrics_dict:
        ab_df = metrics_dict["ablation"]
        ab_display = ab_df.copy()
        name_vn = {
            "calendar_only": "1. Chỉ Lịch (Calendar Only)",
            "lag_rolling_only": "2. Chỉ Lịch sử chuỗi (Lags & Rolling)",
            "full_minus_rolling": "3. Toàn bộ trừ Rolling stats (Lịch + Lags)",
            "full": "4. Toàn bộ đặc trưng (Full Features)",
        }
        ab_display["Nhóm đặc trưng"] = ab_display["config_name"].map(name_vn)
        st.dataframe(ab_display[["Nhóm đặc trưng", "num_features", "mae", "rmse", "r2", "fit_time_seconds"]], width="stretch")

        fig_ab = px.bar(
            ab_df,
            x="config_name",
            y=["mae", "rmse"],
            barmode="group",
            labels={"value": "Sai số (kW)", "config_name": "Cấu hình", "variable": "Chỉ số"},
            title="So sánh sai số giữa các tập đặc trưng trên Validation Set",
            color_discrete_map={"mae": "#e377c2", "rmse": "#c5b0d5"},
        )
        st.plotly_chart(fig_ab, width="stretch")

        # Dynamic delta calculation from table
        cal_row = ab_df[ab_df["config_name"] == "calendar_only"]
        full_row = ab_df[ab_df["config_name"] == "full"]
        if len(cal_row) > 0 and len(full_row) > 0:
            cal_mae = float(cal_row["mae"].iloc[0])
            full_mae = float(full_row["mae"].iloc[0])
            delta_cal = cal_mae - full_mae
            st.markdown(
                f"""
                **Nhận xét số liệu từ thí nghiệm Ablation:**
                - Khi chỉ sử dụng các đặc trưng lịch (`calendar_only`), MAE tăng thêm **+{delta_cal:.4f} kW** so với mô hình đầy đủ (`full`), cho thấy thông tin độ trễ ngắn hạn là yếu tố quyết định chất lượng dự báo một giờ tiếp theo.
                - Sự chênh lệch giữa `full_minus_rolling` và `full` là tương đối nhỏ trên tập dữ liệu này, cho thấy phần lớn tín hiệu tự hồi quy đã được biểu diễn hiệu quả qua các bước trễ chính.
                """
            )

# ==================== TAB 6: METHODOLOGY & LIMITATIONS ====================
with tab6:
    st.header("Phương Pháp Luận & Giới Hạn Thực Tế")
    st.caption("Dữ liệu: Georges Hébrail & Alice Bérard — UCI Individual Household Electric Power Consumption, "
               "[DOI 10.24432/C58885](https://doi.org/10.24432/C58885), "
               "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).")

    st.subheader("1. Kiến trúc xử lý và luồng dữ liệu")
    st.markdown(
        """
        - **Dữ liệu gốc**: 2.075.259 phút đo đạc từ 12/2006 đến 11/2010 (UCI Dataset 235).
        - **Chuẩn hóa tần suất**: Tổng hợp thành chuỗi liên tục 1 giờ theo nhãn trái `[s, s+1h)`.
        - **Tiêu chuẩn tính nhãn hợp lệ**: Một giờ s chỉ được tính là nhãn hợp lệ (y_s) khi có đủ **60 phút đo đạc không bị thiếu**. Không sinh nhãn giả mạo.
        - **Xử lý thiếu khuyết**: Dữ liệu đầu vào lịch sử được điền tiến có chặn (bounded forward fill tối đa 3 giờ), sau đó sử dụng SimpleImputer chỉ fit trên tập huấn luyện của fold.
        - **Phân chia thời gian**:
          - Warmup: 168 giờ đầu được dùng để khởi tạo cửa sổ trễ/trượt, loại bỏ khỏi đánh giá.
          - Train: 70% thời gian đầu (thực hiện 3-fold Expanding Cross-Validation để chọn siêu tham số).
          - Validation: 15% thời gian tiếp theo (dùng để chọn mô hình chiến thắng và cố định cấu hình).
          - Test: 15% thời gian cuối cùng; cố định mô hình trước khi đánh giá. Các lần chạy lại chỉ phục vụ kiểm chứng kỹ thuật.
        """
    )

    st.subheader("2. Khái niệm Công suất (kW) và Điện năng (kWh)")
    st.markdown(
        """
        - **kW (Kilowatt)**: Đơn vị đo công suất tác dụng (tốc độ tiêu thụ năng lượng trung bình trong khoảng 1 giờ).
        - **kWh (Kilowatt-giờ)**: Đơn vị đo điện năng tiêu thụ tích lũy (E = P × t). Với chu kỳ dự báo 1 giờ (t = 1h), công suất trung bình 1.5 kW tương ứng đúng 1.5 kWh điện năng tiêu thụ trong giờ đó.
        """
    )

    st.subheader("3. Giới hạn phạm vi ứng dụng thực tế")
    st.warning(
        """
        Khuyến cáo phạm vi nghiên cứu:
        1. **Dữ liệu của 01 hộ gia đình duy nhất tại Sceaux (Pháp) giai đoạn 2006–2010**: Dữ liệu mang tính chất hành vi cá nhân của hộ gia đình khảo sát tại châu Âu, không đại diện cho toàn bộ các hộ gia đình khác và không phản ánh cơ cấu phụ tải hay thời tiết tại Việt Nam.
        2. **Dự báo không đồng nghĩa với tự động tiết kiệm điện**: Mô hình đóng vai trò cung cấp thông tin dự báo; việc tiết kiệm chi phí chỉ phát huy hiệu quả khi kết hợp với chính sách giá điện theo giờ, hệ thống pin tích trữ, hoặc các giải pháp quản lý phụ tải tự động.
        3. **Quy ước dự báo một bước (One-step Rolling Forecast)**: Mô hình thực hiện dự báo cho 1 giờ tiếp theo với giả định dữ liệu thực tế của các giờ trước được cập nhật tuần tự, không phải là bài toán dự báo mở nhiều ngày liên tiếp.
        """
    )
