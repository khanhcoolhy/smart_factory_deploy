import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import plotly.graph_objects as go
# Bỏ make_subplots để vẽ rời cho nhẹ
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
        print(f"Mail Error: {e}")
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
# 2. XỬ LÝ DỮ LIỆU (Tối ưu bộ nhớ)
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def process_and_enrich(df_input, _config):
    try:
        # Giảm bộ nhớ ngay từ đầu
        if 'data' in df_input.columns:
            # Chỉ lấy cột cần thiết nếu file quá lớn
            pass 

        df_input['time'] = pd.to_datetime(df_input['time'], format='mixed', utc=True)

        frames = []
        unique_devices = df_input['DevAddr'].unique()

        for dev in unique_devices:
            mask = df_input['DevAddr'] == dev
            df_subset = df_input.loc[mask].copy()
            
            # Tìm kênh
            found_channels = [col.split('.')[0] for col in df_subset.columns if col.endswith('.Actual')]
            if not found_channels: continue 
            ch = found_channels[0]
            
            cols_map = {
                f'{ch}.Actual': 'Actual', f'{ch}.Status': 'Status',
                f'{ch}.Actual2': 'Actual2', f'{ch}.RunTime': 'RunTime',
                f'{ch}.HeldTime': 'HeldTime'
            }
            # Chỉ rename cột có thật
            valid_map = {k: v for k, v in cols_map.items() if k in df_subset.columns}
            df_subset.rename(columns=valid_map, inplace=True)
            
            # Fill thiếu
            for c in ['Actual', 'Status', 'RunTime', 'HeldTime', 'Actual2']:
                if c not in df_subset.columns: df_subset[c] = 0
            
            # Chỉ giữ cột cần thiết để tiết kiệm RAM
            keep_cols = ['DevAddr', 'time', 'Actual', 'Status', 'RunTime', 'HeldTime', 'Actual2']
            df_subset = df_subset[keep_cols]
            
            # Cast float32 để nhẹ
            float_cols = ['Actual', 'RunTime', 'HeldTime', 'Actual2']
            df_subset[float_cols] = df_subset[float_cols].astype('float32')
            
            frames.append(df_subset)

        if not frames: return None
        df = pd.concat(frames, ignore_index=True)
        df.sort_values(by=['DevAddr', 'time'], inplace=True)
        
        # Feature Engineering
        grp = df.groupby('DevAddr')
        df['Speed'] = grp['Actual'].diff().fillna(0).astype('float32')
        df['d_RunTime'] = grp['RunTime'].diff().fillna(0).astype('float32')
        df['d_HeldTime'] = grp['HeldTime'].diff().fillna(0).astype('float32')
        
        df = df[(df['Speed'] >= 0) & (df['Speed'] < 50000)].copy()

        # Weather (Giả lập để tránh lỗi mạng gây crash)
        df['Temp'] = 25.0
        df['Humidity'] = 70.0
        
        return df

    except Exception as e:
        st.error(f"Lỗi xử lý: {str(e)}")
        return None

# ==========================================
# 3. GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="Smart Factory AI", layout="wide", page_icon="🏭")
st.title("🏭 Hệ thống Giám sát Nhà máy (Stable Version)")

# Load Model (BỎ QUANTIZATION)
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
        # Map location cpu quan trọng để tránh lỗi CUDA trên Cloud
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        # [FIX CRASH] KHÔNG DÙNG QUANTIZATION NỮA
        return model, scaler, config
    except Exception as e:
        st.error(f"Lỗi load model: {str(e)}")
        return None, None, None

model, scaler, config = load_system_components()

if not model:
    st.error("⚠️ Không tìm thấy Model! Kiểm tra lại file model/config.")
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
st.sidebar.header("📥 Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
COST_PER_ERROR = st.sidebar.number_input("Chi phí lỗi (VND)", value=5000, step=1000)

if uploaded_file:
    if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
        st.session_state.analysis_done = False
        st.session_state.last_file = uploaded_file.name

    # Đọc file (có thể dùng chunksize nếu file quá lớn, ở đây đọc thường)
    df_input = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Load: {len(df_input)} dòng")

    with st.spinner("Đang xử lý dữ liệu..."):
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
            selected_dev = st.selectbox("👉 Chọn thiết bị:", unique_devs)
            if st.session_state.selected_dev != selected_dev:
                st.session_state.analysis_done = False
                st.session_state.selected_dev = selected_dev
                st.session_state.res = None

        with col2:
            st.write("")
            st.write("")
            turbo_mode = st.checkbox("⚡ Turbo Mode", value=True)

        df_machine = df_processed[df_processed['Label'] == selected_dev].sort_values('time')

        with st.expander("🔍 Xem dữ liệu"):
            st.dataframe(df_machine.head(100))

        if len(df_machine) < config['seq_length'] + 5:
            st.warning("Dữ liệu quá ngắn.")
        else:
            if st.button("🚀 BẮT ĐẦU PHÂN TÍCH", type="primary", use_container_width=True):
                try:
                    # 1. Prepare
                    req_cols = config['features_list']
                    # Fix thiếu cột
                    for c in req_cols:
                        if c not in df_machine.columns: df_machine[c] = 0.0
                        
                    data_log = np.log1p(df_machine[req_cols].values)
                    data_vals = scaler.transform(data_log)
                    
                    seq_len = config['seq_length']
                    step_size = 10 if turbo_mode else 1
                    
                    # Tạo sequence bằng numpy stride hoặc loop (loop an toàn hơn về mem)
                    X_list = []
                    valid_indices = []
                    
                    # Giới hạn số lượng điểm nếu quá lớn để tránh OOM
                    MAX_SAMPLES = 50000 
                    if len(data_vals) > MAX_SAMPLES and not turbo_mode:
                         st.warning(f"Dữ liệu lớn ({len(data_vals)} dòng). Tự động bật Turbo để tránh tràn bộ nhớ.")
                         step_size = 10

                    for i in range(0, len(data_vals) - seq_len, step_size):
                        X_list.append(data_vals[i:i+seq_len])
                        valid_indices.append(i + seq_len)
                    
                    if not X_list:
                        st.error("Không đủ dữ liệu.")
                        st.stop()

                    X_input = torch.tensor(np.array(X_list), dtype=torch.float32)
                    
                    # 2. Inference (Batching)
                    dataset = torch.utils.data.TensorDataset(X_input)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
                    
                    all_preds = []
                    prog_bar = st.progress(0, text="AI Analyzing...")
                    
                    with torch.no_grad():
                        for i, batch in enumerate(dataloader):
                            preds = model(batch[0])
                            all_preds.append(preds.numpy())
                            prog_bar.progress(min((i+1)/len(dataloader), 1.0))
                    prog_bar.empty()

                    # 3. Calc Loss
                    predictions = np.concatenate(all_preds, axis=0)
                    actuals = data_vals[valid_indices]
                    
                    target_idx = config.get('target_cols_idx', [0, 1, 2])
                    mae_loss = np.mean(np.abs(predictions[:, target_idx] - actuals[:, target_idx]), axis=1)

                    # 4. Smart Threshold
                    res = df_machine.iloc[valid_indices].copy()
                    
                    # Fix NaN/Inf
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
                    
                    # 5. Logic
                    cond_ai = res['Anomaly_Score'] > final_thresh
                    cond_running = res['Speed'] > 0.1
                    res['Is_Anomaly'] = cond_ai & cond_running

                    st.session_state.res = res
                    st.session_state.n_err = res['Is_Anomaly'].sum()
                    st.session_state.analysis_done = True
                    
                    # 6. Email
                    n_err = st.session_state.n_err
                    loss_vnd = n_err * COST_PER_ERROR
                    status = "CÓ VẤN ĐỀ" if n_err > 0 else "ỔN ĐỊNH"
                    
                    msg = (
                        f"Device: {selected_dev}\n"
                        f"Status: {status}\n"
                        f"Threshold: {final_thresh:.4f} ({best_method})\n"
                        f"Errors: {n_err}\n"
                        f"Est. Loss: {loss_vnd:,.0f} VND"
                    )
                    
                    send_gmail_report(f"AI REPORT: {status} | {selected_dev}", msg)
                    st.toast("Phân tích hoàn tất!", icon="✅")

                except Exception as e:
                    st.error(f"Crash Error: {str(e)}")
                    # Gợi ý fix nếu OOM
                    st.warning("Gợi ý: Nếu file quá lớn, hãy bật 'Chế độ Turbo' hoặc giảm bớt dữ liệu.")

            # --- DISPLAY ---
            if st.session_state.analysis_done and st.session_state.res is not None:
                res = st.session_state.res
                n_err = st.session_state.n_err
                thresh = st.session_state.final_threshold
                method = st.session_state.thresh_method
                
                st.info(f"🧠 **AI Auto-Tuning:** Ngưỡng: **{thresh:.4f}** | Phương pháp: **{method}**")

                k1, k2, k3 = st.columns(3)
                with k1:
                    if n_err == 0: st.success(f"TRẠNG THÁI\n# ỔN ĐỊNH")
                    else: st.error(f"TRẠNG THÁI\n# CÓ LỖI ({n_err})")
                with k2: st.metric("Tỷ lệ lỗi", f"{(n_err/len(res))*100:.2f}%")
                with k3: st.metric("Thiệt hại", f"{n_err * COST_PER_ERROR:,.0f} đ")

                st.divider()
                st.subheader("📊 Biểu đồ chi tiết")
                
                # --- [TỐI ƯU] Downsample dữ liệu vẽ biểu đồ ---
                # Vẽ nhiều điểm quá sẽ làm lag trình duyệt
                df_err = res[res['Is_Anomaly']]
                
                PLOT_LIMIT = 3000 # Giới hạn số điểm vẽ
                if len(res) > PLOT_LIMIT:
                    step = len(res) // PLOT_LIMIT
                    df_viz = res.iloc[::step].copy()
                    # Luôn giữ lại các điểm lỗi để không bị mất khi downsample
                    if not df_err.empty:
                        df_viz = pd.concat([df_viz, df_err]).drop_duplicates(subset=['time']).sort_values('time')
                else:
                    df_viz = res

                # CHART 1: TỐC ĐỘ
                fig_speed = go.Figure()
                fig_speed.add_trace(go.Scattergl(x=df_viz['time'], y=df_viz['Speed'], mode="lines", name="Tốc độ", line=dict(color="#1f77b4", width=1)))
                if not df_err.empty:
                    # Giới hạn số điểm lỗi hiển thị nếu quá nhiều
                    show_err = df_err if len(df_err) < 1000 else df_err.iloc[::len(df_err)//1000]
                    fig_speed.add_trace(go.Scattergl(x=show_err['time'], y=show_err['Speed'], mode="markers", marker=dict(color="red", size=6), name="Lỗi"))
                
                fig_speed.update_layout(title="1. Tốc độ & Điểm lỗi", height=350, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_speed, use_container_width=True)

                # CHART 2: AI SCORE
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scattergl(x=df_viz['time'], y=df_viz['Anomaly_Score'], mode="lines", name="AI Score", line=dict(color="#9467bd", width=1), fill='tozeroy'))
                fig_loss.add_hline(y=thresh, line_dash="dash", line_color="red", annotation_text=f"Ngưỡng: {thresh:.2f}")
                fig_loss.update_layout(title="2. AI Score (Độ lệch chuẩn)", height=300, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_loss, use_container_width=True)

                # CHART 3: MÔI TRƯỜNG
                fig_env = go.Figure()
                fig_env.add_trace(go.Scattergl(x=df_viz['time'], y=df_viz['Temp'], mode="lines", name="Nhiệt độ", line=dict(color="orange", width=1)))
                fig_env.update_layout(title="3. Nhiệt độ môi trường", height=250, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_env, use_container_width=True)
                
                if n_err > 0:
                    with st.expander("📋 Danh sách lỗi (Top 100)"):
                        st.dataframe(
                            res[res['Is_Anomaly']][['time', 'Speed', 'Temp', 'Anomaly_Score']]
                            .sort_values('Anomaly_Score', ascending=False)
                            .head(100), 
                            use_container_width=True
                        )
                
                st.write("") 

    else:
        st.info("👈 Upload file để bắt đầu.")