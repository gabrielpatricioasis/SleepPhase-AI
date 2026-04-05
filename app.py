import streamlit as st
import mne
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="SleepPhase AI | Professional Scoring", layout="wide", page_icon="🌙")

# --- RUTAS Y CARGA DE ARCHIVOS ---
FOLDER_PATH = r'C:\Users\gabri\Documents\CSIM\TM\sleep\subjects'

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('sleep_mlp_model.joblib')
        scaler = joblib.load('sleep_scaler.joblib')
        return model, scaler
    except:
        return None, None

# --- FUNCIONES DE PROCESAMIENTO (Lógica de Eric) ---
def extract_features_from_edf(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
    ch_eeg = [ch for ch in raw.ch_names if 'Fpz-Cz' in ch][0]
    ch_eog = [ch for ch in raw.ch_names if 'EOG horizontal' in ch][0]
    ch_emg = [ch for ch in raw.ch_names if 'EMG submental' in ch][0]
    
    raw.set_channel_types({ch_eeg: 'eeg', ch_eog: 'eog', ch_emg: 'emg'})
    raw.filter(0.5, 30.0, picks=['eeg', 'eog'], verbose=False)
    raw.filter(0.5, 10.0, picks=['emg'], verbose=False)
    raw.resample(100)
    
    epochs = mne.make_fixed_length_epochs(raw, duration=30.0, preload=True, verbose=False)
    
    eeg_data = epochs.get_data(picks='eeg')
    eog_data = epochs.get_data(picks='eog')
    emg_data = epochs.get_data(picks='emg')
    
    psds, freqs = mne.time_frequency.psd_array_multitaper(eeg_data, sfreq=100, fmin=0.5, fmax=30, verbose=False)
    bands = [(0.5, 4), (4, 8), (8, 12), (12, 16), (16, 30)] 
    
    eeg_feat = [[psd[0, np.logical_and(freqs >= f[0], freqs <= f[1])].mean() for f in bands] for psd in psds]
    eog_feat = np.mean(eog_data**2, axis=-1)
    emg_feat = np.mean(np.abs(emg_data), axis=-1)
    
    X = np.hstack([np.array(eeg_feat), eog_feat, emg_feat])
    return X, raw.info['meas_date']

# --- INTERFAZ ---
st.title("🌙 SleepPhase AI: Automated Scoring System")

# Cargar Cerebro
mlp, scaler = load_model_and_scaler()

# Sidebar: Selección de Sujeto y Disclaimer
with st.sidebar:
    st.header("📂 Data Selection")
    
    # Disclaimer científico añadido
    st.info("⚠️ This model is calibrated for the Sleep-EDF (Kemp et al.) database standards. Model accuracy (92.0%) is based on global database validation.")
    
    if os.path.exists(FOLDER_PATH):
        edf_files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith('.edf') and 'psg' in f.lower()]
        selected_file = st.selectbox("Select Subject PSG File", edf_files)
    else:
        st.error(f"Folder not found: {FOLDER_PATH}")
        selected_file = None
    
    analyze_btn = st.button("🚀 Run Automated Labeling")

# 1. Tabla de Referencia (Mejorada sin índices y con nombres correctos)
st.subheader("📋 Physiological Marker Reference")
ref_data = {
    "Stage": ["Wake", "N1", "N2", "N3", "N4 (Classic)", "REM"],
    "EEG Marker": ["Alpha waves", "Theta waves", "Spindles (12-14 Hz) & K-Complexes", "Delta (20-50%)", "Delta (>50%)", "Sawtooth waves"],
    "Clinical Insight": ["Awake", "Light sleep (Hypnagogia)", "Sleep Onset / Memory Consolidation", "Slow Wave Sleep (SWS)", "Deepest Sleep", "Dreaming state"]
}
# Convertir a DataFrame y mostrar escondiendo el índice
df_ref = pd.DataFrame(ref_data)
st.dataframe(df_ref, hide_index=True, use_container_width=True)

# 2. Lógica de Análisis
if analyze_btn and selected_file and mlp:
    with st.spinner(f"Analyzing {selected_file}..."):
        X_raw, start_dt = extract_features_from_edf(os.path.join(FOLDER_PATH, selected_file))
        X_scaled = scaler.transform(X_raw)
        y_pred = mlp.predict(X_scaled)
        
        from scipy.signal import medfilt
        y_pred = medfilt(y_pred, kernel_size=5).astype(int)

        m1, m2, m3 = st.columns(3)
        m1.metric("System Reliability", "92.0%", "MLP Verified")
        m2.metric("Total Duration", f"{len(y_pred)*30/3600:.1f} Hours")
        m3.metric("Status", "Analysis Complete", "✅")

        # 3. Hipnograma Profesional
        st.subheader("📊 Generated Hypnogram (MLP Prediction)")
        
        fig, ax = plt.subplots(figsize=(15, 5), facecolor='white')
        
        config = {
            4: (0.0, 0.2, '#FF69B4', 'REM'),
            3: (0.2, 0.2, '#6C5CE7', 'N3/4'),
            2: (0.4, 0.2, '#A29BFE', 'N2'),
            1: (0.6, 0.2, '#D1D1FF', 'N1'),
            0: (0.8, 0.2, '#FFB347', 'Wake')
        }

        for stage, (y_min, height, color, label) in config.items():
            is_stage = (y_pred == stage).astype(int)
            diff = np.diff(np.concatenate([[0], is_stage, [0]]))
            starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
            
            for s, e in zip(starts, ends):
                dt_s = start_dt + timedelta(seconds=int(s*30))
                dt_e = start_dt + timedelta(seconds=int(e*30))
                s_num, w_num = mdates.date2num(dt_s), mdates.date2num(dt_e) - mdates.date2num(dt_s)
                ax.broken_barh([(s_num, w_num)], (y_min, height), facecolors=color, edgecolors='none')

        ax.set_ylim(0, 1)
        ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.set_yticklabels(['REM', 'N3/4', 'N2', 'N1', 'Wake'], fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.grid(axis='x', linestyle=':', alpha=0.3)
        st.pyplot(fig)

elif analyze_btn and not mlp:
    st.error("Missing model files! Please ensure 'sleep_mlp_model.joblib' and 'sleep_scaler.joblib' are in the same folder as app.py.")