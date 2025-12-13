import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import plotly.graph_objects as go
# Không cần make_subplots nữa vì ta sẽ vẽ riêng lẻ
import openmeteo_requests
import requests_cache
from retry_requests import retry
import os
import gc
import smtplib 
import ssl 
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ===============================================================
# 0. CẤU HÌNH GMAIL & UTILS
# ===============================================================
def get_gmail_secrets():
    try:
        user = st.secrets.get("GMAIL_USER") or os.environ.get("GMAIL_USER")
        password = st.secrets.get("GMAIL_PASSWORD") or os.environ.get("GMAIL_PASSWORD")
        receiver = st.secrets.get("RECEIVER_EMAIL") or os.environ.get("RECEIVER_EMAIL")
        return user, password, receiver
    except Exception:
        return os.environ.get("GMAIL_USER"), os.environ.get("GMAIL_PASSWORD"), os.environ.get("RECEIVER_EMAIL")

def send_gmail_report(subject, message):
    sender_email, password, receiver_email = get_gmail_secrets()
    if not sender_email or not password or not receiver_email:
        print("⚠️ Thiếu cấu hình Gmail. Bỏ qua gửi email.")
        return False
        
    msg = MIMEMultipart("alternative")
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    
    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif;">
        <h3 style="color: #d9534f;">🚨 BÁO CÁO CẢNH BÁO TỰ ĐỘNG</h3>
        <div style="background-color: #f9f9f9; padding: 15px; border-left: 5px solid #d9534f;">
            <pre style="font-family: monospace; white-space: pre-wrap;">{message}</pre>
        </div>
        <p style="font-size: 12px; color: gray; margin-top: 20px;">
            Hệ thống Smart Factory AI (Adaptive Threshold).
        </p>
      </body>
    </html>
    """
    msg.attach(MIMEText(html_content, "html"))
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            return True
    except Exception as e:
        print(f"❌ Lỗi gửi email: {e}")
        return False

# ==========================================
# 1. CẤU TRÚC MODEL LSTM
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
# 2. XỬ LÝ DỮ LIỆU
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def process_and_enrich(df_input, _config):
    try:
        if 'data' in df_input.columns:
            def parse_safe(x):
                try: return json.loads(str(x).replace("'", "\""))
                except: return {}
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
            valid_cols = [c for c in cols_map.keys() if c in df_subset.columns]
            if not valid_cols: continue
            df_subset.rename(columns=cols_map, inplace=True)
            
            required_cols = ['DevAddr', 'time', 'Actual', 'Status', 'Actual2', 'RunTime', 'HeldTime']
            for c in required_cols:
                if c not in df_subset.columns: df_subset[c] = 0
            
            df_subset = df_subset[required_cols]
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

        # Weather API
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
# 3. GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="Smart Factory AI", layout="wide", page_icon="🏭")
st.title("🏭 Hệ thống Giám sát Nhà máy thông minh (Smart AI)")

# Load Model
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
    st.error("⚠️ Không tìm thấy Model!")
    st.stop()

# Session State
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.res = None
    st.session_state.n_err = 0
    st.session_state.final_threshold = 0.0
    st.session_state.thresh_method = ""
    st.session_state.selected_dev = None 

# Sidebar
st.sidebar.header("📥 Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Chọn file CSV", type=["csv"])
COST_PER_ERROR = st.sidebar.number_input("Đơn giá thiệt hại/lỗi (VND)", value=5000, step=1000)

if uploaded_file:
    if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
        st.session_state.analysis_done = False
        st.session_state.last_file = uploaded_file.name

    df_input = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Đã tải: {len(df_input):,} dòng")

    with st.spinner("🔄 Đang xử lý dữ liệu..."):
        df_processed = process_and_enrich(df_input, config)

    if df_processed is not None and not df_processed.empty:
        # Labeling
        unique_dev_raw = df_processed['DevAddr'].unique()
        dev_map = {dev: f"{dev}_{i+1:02d}" for i, dev in enumerate(unique_dev_raw)}
        df_processed['Label'] = df_processed['DevAddr'].map(dev_map)
        unique_devs = df_processed['Label'].unique()

        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_dev = st.selectbox("👉 **Chọn thiết bị:**", unique_devs)
            if st.session_state.selected_dev != selected_dev:
                st.session_state.analysis_done = False
                st.session_state.selected_dev = selected_dev
                st.session_state.res = None

        with col2:
            st.write("")
            st.write("")
            turbo_mode = st.checkbox("⚡ Chế độ Turbo", value=True)

        df_machine = df_processed[df_processed['Label'] == selected_dev].sort_values('time')

        with st.expander("🔍 Xem dữ liệu thô"):
            st.dataframe(df_machine.head(100))

        if len(df_machine) < config['seq_length'] + 5:
            st.warning(f"⚠️ Dữ liệu quá ngắn.")
        else:
            if st.button("🚀 BẮT ĐẦU PHÂN TÍCH", type="primary", use_container_width=True):
                try:
                    # 1. Prepare Data
                    req_cols = config['features_list']
                    for c in req_cols:
                        if c not in df_machine.columns: df_machine[c] = 0
                        
                    data_log = np.log1p(df_machine[req_cols])
                    data_vals = scaler.transform(data_log)
                    
                    seq_len = config['seq_length']
                    step_size = 10 if turbo_mode else 1
                    indexes = range(0, len(data_vals) - seq_len, step_size)
                    sequences = [data_vals[i:i+seq_len] for i in indexes]

                    if not sequences:
                        st.error("Lỗi tạo sequence.")
                        st.stop()

                    X_input = torch.tensor(np.array(sequences), dtype=torch.float32)
                    dataset = torch.utils.data.TensorDataset(X_input)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False)

                    # 2. Run Model
                    all_preds = []
                    prog_bar = st.progress(0, text="🤖 AI đang phân tích...")
                    with torch.no_grad():
                        for i, batch in enumerate(dataloader):
                            preds = model(batch[0])
                            all_preds.append(preds.numpy())
                            prog_bar.progress(min((i+1)/len(dataloader), 1.0))
                    prog_bar.empty()

                    # 3. Calc Loss
                    predictions = np.concatenate(all_preds, axis=0)
                    actual_indices = [i + seq_len for i in indexes]
                    actuals = data_vals[actual_indices]
                    
                    target_idx = config.get('target_cols_idx', [0, 1, 2])
                    mae_loss = np.mean(np.abs(predictions[:, target_idx] - actuals[:, target_idx]), axis=1)

                    # 4. === SMART AUTO-THRESHOLDING ===
                    res = df_machine.iloc[actual_indices].copy()
                    
                    ai_loss_safe = np.nan_to_num(mae_loss, nan=0.0, posinf=0.0, neginf=0.0)
                    res['Anomaly_Score'] = ai_loss_safe.astype('float32')
                    
                    running_mask = res['Speed'] > 0.0
                    
                    if running_mask.sum() > 50:
                        loss_run = res.loc[running_mask, 'Anomaly_Score'].values
                        mean = np.mean(loss_run)
                        std = np.std(loss_run)
                        th_sigma = mean + 3 * std
                        Q1 = np.nanpercentile(loss_run, 25)
                        Q3 = np.nanpercentile(loss_run, 75)
                        th_iqr = Q3 + 3 * (Q3 - Q1)
                        th_perc = np.nanpercentile(loss_run, 99.5)
                        
                        candidates = {'3-Sigma': th_sigma, 'IQR': th_iqr, 'Percentile': th_perc}
                        best_method = max(candidates, key=candidates.get)
                        final_thresh = float(candidates[best_method])
                        final_thresh = max(final_thresh, 1.0)
                    else:
                        final_thresh = 2.0
                        best_method = "Default"

                    st.session_state.final_threshold = final_thresh
                    st.session_state.thresh_method = best_method
                    
                    # 5. Apply Logic
                    cond_ai = res['Anomaly_Score'] > final_thresh
                    cond_running = res['Speed'] > 0.1
                    res['Is_Anomaly'] = cond_ai & cond_running

                    st.session_state.res = res
                    st.session_state.n_err = res['Is_Anomaly'].sum()
                    st.session_state.analysis_done = True
                    
                    # 6. Email Report
                    n_err = st.session_state.n_err
                    loss_vnd = n_err * COST_PER_ERROR
                    status = "CÓ VẤN ĐỀ" if n_err > 0 else "ỔN ĐỊNH"
                    
                    msg = (
                        f"Thiết bị: {selected_dev}\n"
                        f"Trạng thái: {status}\n"
                        f"Ngưỡng áp dụng: {final_thresh:.4f} ({best_method})\n"
                        f"Số lỗi: {n_err}\n"
                        f"Thiệt hại: {loss_vnd:,.0f} VND"
                    )
                    
                    with st.spinner("Đang gửi email báo cáo..."):
                        if send_gmail_report(f"AI REPORT: {status} | {selected_dev}", msg):
                            st.toast(f"Đã gửi báo cáo qua Email! (Ngưỡng: {final_thresh:.2f})", icon="📧")

                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")

            # --- DISPLAY RESULTS ---
            if st.session_state.analysis_done and st.session_state.res is not None:
                res = st.session_state.res
                n_err = st.session_state.n_err
                thresh = st.session_state.final_threshold
                method = st.session_state.thresh_method
                
                # --- [FIX GIAO DIỆN] Tách biểu đồ thành 3 chart riêng biệt ---
                # --- Để tránh lỗi layout bị kẹt không cuộn được ---
                
                st.info(f"🧠 **AI Auto-Tuning:** Ngưỡng chốt: **{thresh:.4f}** | Phương pháp: **{method}**")

                k1, k2, k3 = st.columns(3)
                with k1:
                    if n_err == 0: st.success(f"TRẠNG THÁI\n# ỔN ĐỊNH ✅")
                    else: st.error(f"TRẠNG THÁI\n# CÓ LỖI 🚨 ({n_err})")
                with k2: st.metric("Tỷ lệ lỗi", f"{(n_err/len(res))*100:.2f}%")
                with k3: st.metric("Thiệt hại ước tính", f"{n_err * COST_PER_ERROR:,.0f} đ")

                st.divider()
                st.subheader("📊 Biểu đồ chi tiết")
                
                # Chuẩn bị data vẽ
                df_err = res[res['Is_Anomaly']]
                MAX_POINTS = 5000 # Giảm bớt điểm vẽ để mượt hơn
                if len(res) > MAX_POINTS:
                    step = len(res) // MAX_POINTS
                    df_viz = res.iloc[::step].copy()
                    df_viz = pd.concat([df_viz, df_err]).drop_duplicates(subset=['time']).sort_values('time')
                else:
                    df_viz = res

                # --- BIỂU ĐỒ 1: TỐC ĐỘ ---
                fig_speed = go.Figure()
                fig_speed.add_trace(go.Scattergl(x=df_viz['time'], y=df_viz['Speed'], mode="lines", name="Tốc độ", line=dict(color="#1f77b4")))
                if not df_err.empty:
                    fig_speed.add_trace(go.Scattergl(x=df_err['time'], y=df_err['Speed'], mode="markers", marker=dict(color="red", size=8, symbol="x"), name="Lỗi"))
                
                # Auto zoom Y
                ymax = df_viz['Speed'].quantile(0.99) * 1.5
                if ymax > 0: fig_speed.update_yaxes(range=[0, ymax])
                fig_speed.update_layout(title="1. Hoạt động thực tế (Tốc độ)", height=350, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_speed, use_container_width=True)

                # --- BIỂU ĐỒ 2: AI SCORE ---
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scattergl(x=df_viz['time'], y=df_viz['Anomaly_Score'], mode="lines", name="AI Score", line=dict(color="#9467bd"), fill='tozeroy'))
                fig_loss.add_hline(y=thresh, line_dash="dash", line_color="red", annotation_text=f"Ngưỡng: {thresh:.2f}")
                fig_loss.update_layout(title="2. Điểm số bất thường của AI (Càng cao càng lỗi)", height=300, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_loss, use_container_width=True)

                # --- BIỂU ĐỒ 3: MÔI TRƯỜNG ---
                fig_env = go.Figure()
                fig_env.add_trace(go.Scattergl(x=df_viz['time'], y=df_viz['Temp'], mode="lines", name="Nhiệt độ", line=dict(color="orange")))
                fig_env.update_layout(title="3. Môi trường (Nhiệt độ)", height=300, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_env, use_container_width=True)
                
                # BẢNG CHI TIẾT
                if n_err > 0:
                    st.divider()
                    st.subheader("📋 Danh sách điểm lỗi")
                    st.dataframe(
                        res[res['Is_Anomaly']][['time', 'Speed', 'Temp', 'Anomaly_Score']]
                        .sort_values('Anomaly_Score', ascending=False)
                        .head(500), 
                        use_container_width=True
                    )
                
                # Spacer cuối cùng để đảm bảo scroll được hết
                st.write("")
                st.write("")

    else:
        st.info("👈 Vui lòng upload file CSV.")