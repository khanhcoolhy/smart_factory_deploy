import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openmeteo_requests
import requests_cache
from retry_requests import retry
import os
import gc
import requests
# THƯ VIỆN MỚI CHO GMAIL
import smtplib 
import ssl 
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ===============================================================
# 0. CẤU HÌNH GMAIL (SỬ DỤNG MẬT KHẨU ỨNG DỤNG)
# ===============================================================

# Hàm này sẽ được gọi bên trong send_gmail_report để đảm bảo đọc được Secrets
def get_gmail_secrets():
    """Đọc cấu hình Gmail (User, Password, Receiver) từ Streamlit/Hugging Face Secrets."""
    # Trên Hugging Face, bạn cần cấu hình GMAIL_USER, GMAIL_PASSWORD, RECEIVER_EMAIL
    try:
        # Nếu đang chạy trên Streamlit/Hugging Face Spaces, dùng st.secrets
        # Nếu chạy local, dùng os.environ (hoặc đọc từ .streamlit/secrets.toml qua st.secrets)
        user = st.secrets.get("GMAIL_USER") or os.environ.get("GMAIL_USER")
        password = st.secrets.get("GMAIL_PASSWORD") or os.environ.get("GMAIL_PASSWORD")
        receiver = st.secrets.get("RECEIVER_EMAIL") or os.environ.get("RECEIVER_EMAIL")
             
        return user, password, receiver
    except Exception as e:
        # Fallback an toàn (chỉ lấy từ os.environ)
        return os.environ.get("GMAIL_USER"), os.environ.get("GMAIL_PASSWORD"), os.environ.get("RECEIVER_EMAIL")


def send_gmail_report(subject, message):
    """Hàm gửi báo cáo về Gmail."""
    
    sender_email, password, receiver_email = get_gmail_secrets()
    
    # 1. KIỂM TRA SECRETS CÓ ĐƯỢC LOAD KHÔNG
    if not sender_email or not password or not receiver_email:
        print("⚠️ Cảnh báo: Thiếu biến môi trường GMAIL. Bỏ qua gửi báo cáo.")
        st.toast("⚠️ Lỗi: Không thể gửi báo cáo (Thiếu Gmail Secrets).", icon="❌")
        return False
        
    # Cấu hình SMTP
    smtp_server = "smtp.gmail.com"
    port = 465  # Cổng SSL
    
    # Tạo nội dung Email
    msg = MIMEMultipart("alternative")
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    
    # Chuyển Markdown sang HTML đơn giản để hiển thị đẹp hơn trong email
    html_content = f"""\
    <html>
      <body>
        <p style="font-family: monospace;">AI Smart Factory Report</p>
        <pre style="font-family: monospace;">{message}</pre>
        <p style="font-family: monospace;">Vui lòng kiểm tra ứng dụng Streamlit để xem biểu đồ chi tiết.</p>
      </body>
    </html>
    """
    part1 = MIMEText(html_content, "html")
    msg.attach(part1)
    
    # 2. KIỂM TRA VÀ GỬI
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            print("✅ Gửi báo cáo Gmail thành công!")
            st.toast(f"✅ Đã gửi báo cáo tự động đến {receiver_email}!", icon="📧")
            return True
            
    except smtplib.SMTPAuthenticationError:
        error_msg = "❌ Lỗi xác thực Gmail. Vui lòng kiểm tra lại GMAIL_PASSWORD (cần là Mật khẩu Ứng dụng 16 ký tự)."
        print(error_msg)
        st.toast(error_msg, icon="❌")
        return False
    except Exception as e:
        error_msg = f"❌ Lỗi mạng/SMTP khi kết nối Gmail: {e}"
        print(error_msg)
        st.toast("❌ Lỗi mạng: Không thể kết nối tới SMTP Server. Vui lòng kiểm tra kết nối mạng/VPN/Firewall.", icon="❌")
        return False

# ==========================================
# 1. CẤU TRÚC MODEL (GIỮ NGUYÊN)
# ==========================================
class LSTMPredictor(nn.Module):
    def __init__(self, n_features, hidden_dim=128):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        prediction = self.fc(last_step)
        return prediction

# ==========================================
# 2. HÀM XỬ LÝ DỮ LIỆU (GIỮ NGUYÊN)
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def process_and_enrich(df_input, _config):
    try:
        if 'data' in df_input.columns:
            def parse_safe(x):
                try:
                    return json.loads(str(x).replace("'", "\""))
                except:
                    return {}
            json_list = [parse_safe(x) for x in df_input['data']]
            json_df = pd.json_normalize(json_list)
            df_input = pd.concat([df_input[['DevAddr', 'time']], json_df], axis=1)
            del json_list, json_df
            gc.collect()

        df_input['time'] = pd.to_datetime(df_input['time'], format='mixed', utc=True)

        frames = []
        unique_devices = df_input['DevAddr'].unique()

        for dev in unique_devices:
            mask = df_input['DevAddr'] == dev
            df_subset = df_input.loc[mask].copy()
            found_channels = [col.split('.')[0] for col in df_subset.columns if col.endswith('.Actual')]
            if not found_channels: continue 
            ch = found_channels[0]
            cols_map = {
                f'{ch}.Actual': 'Actual', f'{ch}.Status': 'Status',
                f'{ch}.Actual2': 'Actual2', f'{ch}.RunTime': 'RunTime',
                f'{ch}.HeldTime': 'HeldTime'
            }
            available_cols = set(cols_map.keys()).intersection(df_subset.columns)
            if not available_cols: continue
            df_subset.rename(columns=cols_map, inplace=True)
            keep_cols = ['DevAddr', 'time', 'Actual', 'Status', 'Actual2', 'RunTime', 'HeldTime']
            for c in keep_cols:
                if c not in df_subset.columns: df_subset[c] = 0
            df_subset = df_subset[keep_cols]
            df_subset['Channel'] = ch 
            float_cols = ['Actual', 'Actual2', 'RunTime', 'HeldTime']
            df_subset[float_cols] = df_subset[float_cols].astype('float32')
            frames.append(df_subset)

        if not frames: return None
        df = pd.concat(frames, ignore_index=True)
        df.sort_values(by=['DevAddr', 'time'], inplace=True)
        grp = df.groupby('DevAddr')
        df['Speed'] = grp['Actual'].diff().fillna(0).astype('float32')
        df['d_RunTime'] = grp['RunTime'].diff().fillna(0).astype('float32')
        df['d_HeldTime'] = grp['HeldTime'].diff().fillna(0).astype('float32')
        df = df[(df['Speed'] >= 0) & (df['Speed'] < 50000)].copy()

        if df.empty: return df
        min_date = df['time'].min().strftime('%Y-%m-%d')
        max_date = df['time'].max().strftime('%Y-%m-%d')

        try:
            cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
            retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
            openmeteo = openmeteo_requests.Client(session=retry_session)
            params = {
                "latitude": 21.02, "longitude": 105.83,
                "start_date": min_date, "end_date": max_date,
                "hourly": ["temperature_2m", "relative_humidity_2m"]
            }
            responses = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
            hourly = responses[0].Hourly()
            times = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )
            df_weather = pd.DataFrame({
                "time": times,
                "Temp": hourly.Variables(0).ValuesAsNumpy().astype('float32'),
                "Humidity": hourly.Variables(1).ValuesAsNumpy().astype('float32')
            })
            df_final = pd.merge_asof(df.sort_values('time'), df_weather, on='time', direction='backward')
            df_final[['Temp', 'Humidity']] = df_final[['Temp', 'Humidity']].ffill().bfill()
            return df_final
        except:
            df['Temp'] = 25.0
            df['Humidity'] = 70.0
            return df
    except Exception as e:
        st.error(f"Lỗi xử lý dữ liệu: {str(e)}")
        return None

# ==========================================
# 3. GIAO DIỆN CHÍNH (ĐÃ BỎ NÚT KIỂM TRA TELEGRAM)
# ==========================================
st.set_page_config(page_title="Smart Factory AI", layout="wide", page_icon="🏭")
st.title("🏭 Hệ thống Giám sát Nhà máy thông minh (AI Powered)")

@st.cache_resource
def load_system_components():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "model_config_v2.pkl")
        scaler_path = os.path.join(current_dir, "robust_scaler_v2.pkl")
        model_path = os.path.join(current_dir, "lstm_factory_v2.pth")

        if not os.path.exists(config_path): return None, None, None
        config = joblib.load(config_path)
        scaler = joblib.load(scaler_path)
        model = LSTMPredictor(n_features=config['n_features'], hidden_dim=config['hidden_dim'])
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
        )
        return quantized_model, scaler, config
    except Exception as e:
        st.error(f"Lỗi load model: {str(e)}")
        return None, None, None

model, scaler, config = load_system_components()

if not model:
    st.error("⚠️ **Không tìm thấy Model!** Vui lòng kiểm tra file model, config và scaler.")
    st.info("💡 Lưu ý: Hãy chắc chắn bạn đã upload 3 file: `model_config_v2.pkl`, `robust_scaler_v2.pkl`, và `lstm_factory_v2.pth` lên cùng thư mục với file app.py")
    st.stop()

if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.res = None
    st.session_state.n_err = 0
    st.session_state.selected_dev = None

st.sidebar.header("📥 Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Chọn file CSV dữ liệu máy", type=["csv"])

if uploaded_file:
    if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
        st.session_state.analysis_done = False
        st.session_state.last_file = uploaded_file.name

    df_input = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Đã tải: {len(df_input):,} dòng")
    
    # ĐÃ BỎ NÚT KIỂM TRA TELEGRAM TẠI ĐÂY

    with st.spinner("🔄 Đang chuẩn hóa dữ liệu..."):
        df_processed = process_and_enrich(df_input, config)

    if df_processed is not None and not df_processed.empty:
        def get_fixed_label(row):
            dev_id = row['DevAddr']
            original_ch = row['Channel']
            if dev_id == "4417930D77DA": return "4417930D77DA (Kênh 01)"
            elif dev_id == "AC0BFBCE8797": return "AC0BFBCE8797 (Kênh 02)"
            else: return f"{dev_id} (Kênh {original_ch})"

        df_processed['Label'] = df_processed.apply(get_fixed_label, axis=1)
        unique_devs = df_processed['Label'].unique()

        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_dev = st.selectbox("👉 **Chọn thiết bị cần giám sát:**", unique_devs)
            if st.session_state.selected_dev != selected_dev:
                st.session_state.analysis_done = False
                st.session_state.selected_dev = selected_dev
                st.session_state.res = None

        with col2:
            st.write("")
            st.write("")
            turbo_mode = st.checkbox("⚡ Chế độ Turbo (Nhanh)", value=True)

        df_machine = df_processed[df_processed['Label'] == selected_dev].sort_values('time')

        with st.expander("🔍 Xem dữ liệu thô sau khi xử lý"):
            st.dataframe(df_machine.head(100))

        if len(df_machine) < config['seq_length'] + 5:
            st.warning(f"⚠️ Dữ liệu quá ngắn. Cần tối thiểu {config['seq_length']} dòng.")
        else:
            # ---------------------------------------------------------
            # PHẦN NÚT BẤM VÀ XỬ LÝ TỰ ĐỘNG
            # ---------------------------------------------------------
            if st.button("🚀 BẮT ĐẦU PHÂN TÍCH", type="primary", use_container_width=True):
                try:
                    # 1. Chuẩn bị dữ liệu
                    req_cols = config['features_list']
                    data_log = np.log1p(df_machine[req_cols])
                    data_vals = scaler.transform(data_log)
                    
                    seq_len = config['seq_length']
                    step_size = 10 if turbo_mode else 1
                    indexes = range(0, len(data_vals) - seq_len, step_size)
                    sequences = [data_vals[i:i+seq_len] for i in indexes]

                    if not sequences:
                        st.error("Không tạo được sequence dữ liệu.")
                        st.stop()

                    X_input = torch.tensor(np.array(sequences), dtype=torch.float32)
                    dataset = torch.utils.data.TensorDataset(X_input)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False)

                    # 2. Chạy AI Model
                    all_preds = []
                    prog_bar = st.progress(0, text="🤖 AI đang phân tích hành vi máy...")
                    with torch.no_grad():
                        for i, batch in enumerate(dataloader):
                            preds = model(batch[0])
                            all_preds.append(preds.numpy())
                            prog_bar.progress(min((i+1)/len(dataloader), 1.0))
                    prog_bar.empty()

                    # 3. Tính toán kết quả
                    predictions = np.concatenate(all_preds, axis=0)
                    actual_indices = [i + seq_len for i in indexes]
                    actuals = data_vals[actual_indices]
                    target_idx = config.get('target_cols_idx', [0, 1, 2])
                    mae_loss = np.mean(np.abs(predictions[:, target_idx] - actuals[:, target_idx]), axis=1)

                    res = df_machine.iloc[actual_indices].copy()
                    res['Anomaly_Score'] = mae_loss.astype('float32')
                    res['Is_Anomaly'] = res['Anomaly_Score'] > config['threshold']

                    st.session_state.res = res
                    st.session_state.n_err = res['Is_Anomaly'].sum()
                    st.session_state.analysis_done = True
                    
                    # ======================================================
                    # 🔥 [AUTO SEND] TỰ ĐỘNG GỬI GMAIL TẠI ĐÂY 🔥
                    # ======================================================
                    n_err = st.session_state.n_err
                    loss_vnd = n_err * 200000 
                    
                    if n_err > 0:
                        status_icon = "🚨"
                        status_text = "CÓ VẤN ĐỀ"
                    else:
                        status_icon = "✅"
                        status_text = "ỔN ĐỊNH"

                    # Tạo nội dung báo cáo (Markdown)
                    report_msg = (
                        f"{status_icon} **BÁO CÁO TỰ ĐỘNG**\n"
                        f"-----------------------------\n"
                        f"📁 File: `{st.session_state.last_file}`\n"
                        f"🤖 Thiết bị: `{selected_dev}`\n"
                        f"📊 Trạng thái: *{status_text}*\n"
                        f"⚠️ Số lỗi phát hiện: `{n_err}`\n"
                        f"📉 Tỷ lệ lỗi: `{(n_err/len(res))*100:.2f}%`\n"
                        f"💸 Thiệt hại ước tính: `{loss_vnd:,.0f} VND`\n"
                        f"-----------------------------\n"
                        f"👉 AI vừa phân tích xong lúc này."
                    )
                    
                    # Thêm Subject cho Email
                    report_subject = f"{status_icon} BÁO CÁO AI: {status_text} | {selected_dev}"
                    
                    with st.spinner("Đang gửi báo cáo về Gmail..."):
                        send_gmail_report(report_subject, report_msg) 
                        st.toast("Đã tự động gửi báo cáo về Gmail!", icon="🚀")
                    # ======================================================

                except Exception as e:
                    st.error(f"Lỗi trong quá trình phân tích: {str(e)}")

            # ---------------------------------------------------------
            # HIỂN THỊ KẾT QUẢ
            # ---------------------------------------------------------
            if st.session_state.analysis_done and st.session_state.res is not None:
                res = st.session_state.res
                n_err = st.session_state.n_err
                st.success(f"✅ Đã phân tích xong {len(res):,} điểm dữ liệu.")

                kpi1, kpi2, kpi3 = st.columns(3)
                with kpi1:
                    if n_err == 0: st.success("### TRẠNG THÁI\n# ỔN ĐỊNH ✅")
                    elif n_err < len(res) * 0.05: st.warning(f"### CẢNH BÁO ⚠️\n# {n_err} bất thường")
                    else: st.error(f"### NGUY HIỂM 🚨\n# {n_err} bất thường")
                
                with kpi2: st.metric("Tỷ lệ lỗi", f"{(n_err/len(res))*100:.2f}%")
                with kpi3:
                    loss_vnd = n_err * 200000 
                    st.metric("Thiệt hại ước tính", f"{loss_vnd:,.0f} đ", delta="- Lãng phí" if n_err > 0 else "Tối ưu", delta_color="inverse")

                st.divider()
                st.subheader("📊 Biểu đồ chi tiết")
                
                MAX_POINTS = 5000
                if len(res) > MAX_POINTS:
                    step = len(res) // MAX_POINTS
                    df_viz = res.iloc[::step]
                else:
                    df_viz = res
                df_err = res[res['Is_Anomaly']]

                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=("Tốc độ máy & Điểm bất thường", "Nhiệt độ & Độ ẩm"),
                    row_heights=[0.6, 0.4]
                )
                fig.add_trace(go.Scattergl(x=df_viz['time'], y=df_viz['Speed'], mode="lines", line=dict(color="#1f77b4", width=1.5), name="Tốc độ"), row=1, col=1)
                if not df_err.empty:
                    fig.add_trace(go.Scattergl(x=df_err['time'], y=df_err['Speed'], mode="markers", marker=dict(color="red", size=8), name="❗ Lỗi"), row=1, col=1)
                fig.add_trace(go.Scattergl(x=df_viz['time'], y=df_viz['Temp'], mode="lines", line=dict(color="#ff7f0e", width=1.5), name="Nhiệt độ"), row=2, col=1)
                fig.add_trace(go.Scattergl(x=df_viz['time'], y=df_viz['Humidity'], mode="lines", line=dict(color="#2ca02c", width=1.5, dash="dot"), name="Độ ẩm"), row=2, col=1)
                fig.update_layout(height=700, hovermode="x unified", legend=dict(orientation="h", y=1.02))
                st.plotly_chart(fig, use_container_width=True)

                if n_err > 0:
                    with st.expander("📋 Xem danh sách lỗi"):
                        st.dataframe(res[res["Is_Anomaly"]][["time", "Speed", "Temp", "Anomaly_Score"]].sort_values("Anomaly_Score", ascending=False), use_container_width=True)
    else:
        st.info("👈 Vui lòng upload file CSV để bắt đầu.")