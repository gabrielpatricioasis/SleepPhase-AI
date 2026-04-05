import streamlit as st
import mne
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy.signal import medfilt
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="SleepPhase-AI | Automated Scoring", layout="wide", page_icon="🌙")

# --- RUTAS ---
FOLDER_PATH = 'subjects'

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('sleep_mlp_model.joblib')
        scaler = joblib.load('sleep_scaler.joblib')
        return model, scaler
    except:
        return None, None

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

# --- SIDEBAR: INFO Y CRÉDITOS DISCRETOS ---
with st.sidebar:
    st.info("⚠️ This model is calibrated for the Sleep-EDF (Kemp et al.) database standards. Model accuracy (92.0%) is based on global database validation.")
    st.markdown("---")
    # Créditos discretos en formato caption (texto pequeño y gris)
    st.caption("Developed by Gabriel Asís-Sagrado, Èric Domingo Roca & Michaela Freire Griffith")
    st.caption("Master's is Cognitive Systems and Interactive Media (UPF) ― 2026")

# --- INTERFAZ ---
st.title("🌙 SleepPhase AI: Automated Scoring System")
mlp, scaler = load_model_and_scaler()

# --- TABS ---
tab1, tab2 = st.tabs(["📂 Real Patient Analysis", "🧪 Hypnogram Simulator"])

# ==========================================
# PESTAÑA 1: ANÁLISIS REAL
# ==========================================
with tab1:
    col_main, col_side = st.columns([3, 1])
    
    with col_side:
        if os.path.exists(FOLDER_PATH):
            edf_files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith('.edf') and 'psg' in f.lower()]
            selected_file = st.selectbox("Select Subject PSG File", edf_files)
        else:
            st.error("Folder 'subjects' not found. Please upload it to GitHub.")
            selected_file = None
        analyze_btn = st.button("🚀 Run Automated Labeling")

    with col_main:
        st.subheader("📋 Physiological Marker Reference")
        ref_data = {
            "Stage": ["Wake", "N1", "N2", "N3", "N4 (Classic)", "REM"],
            "EEG Marker": ["Alpha waves", "Theta waves", "Spindles (12-14 Hz) & K-Complexes", "Delta (20-50%)", "Delta (>50%)", "Sawtooth waves"],
            "Clinical Insight": ["Awake", "Light sleep (Hypnagogia)", "Sleep Onset / Memory Consolidation", "Slow Wave Sleep (SWS)", "Deepest Sleep", "Dreaming state"]
        }
        st.dataframe(pd.DataFrame(ref_data), hide_index=True, use_container_width=True)

    if analyze_btn and selected_file and mlp:
        with st.spinner(f"Analyzing {selected_file}..."):
            X_raw, start_dt = extract_features_from_edf(os.path.join(FOLDER_PATH, selected_file))
            X_scaled = scaler.transform(X_raw)
            y_pred = medfilt(mlp.predict(X_scaled), kernel_size=5).astype(int)

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("System Reliability", "92.0%", "MLP Verified")
            m2.metric("Total Duration", f"{len(y_pred)*30/3600:.1f} Hours")
            m3.metric("Status", "Analysis Complete", "✅")

            st.subheader("📊 Generated Hypnogram (Real Data)")
            fig, ax = plt.subplots(figsize=(15, 5), facecolor='white')
            config_real = {4: (0.0, 0.2, '#FF69B4', 'REM'), 3: (0.2, 0.2, '#6C5CE7', 'N3/4'),
                           2: (0.4, 0.2, '#A29BFE', 'N2'), 1: (0.6, 0.2, '#D1D1FF', 'N1'), 0: (0.8, 0.2, '#FFB347', 'Wake')}
            
            for stage, (y_min, height, color, label) in config_real.items():
                is_stage = (y_pred == stage).astype(int)
                diff = np.diff(np.concatenate([[0], is_stage, [0]]))
                starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
                for s, e in zip(starts, ends):
                    dt_s = start_dt + timedelta(seconds=int(s*30))
                    dt_e = start_dt + timedelta(seconds=int(e*30))
                    ax.broken_barh([(mdates.date2num(dt_s), mdates.date2num(dt_e) - mdates.date2num(dt_s))], (y_min, height), facecolors=color, edgecolors='none')

            ax.set_ylim(0, 1)
            ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
            ax.set_yticklabels(['REM', 'N3/4', 'N2', 'N1', 'Wake'], fontweight='bold')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.grid(axis='x', linestyle=':', alpha=0.3)
            st.pyplot(fig)
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
            st.download_button(
                label="💾 Download Real Hypnogram",
                data=buf.getvalue(),
                file_name=f"Hypnogram_{selected_file}.png",
                mime="image/png"
            )

# ==========================================
# PESTAÑA 2: SIMULADOR
# ==========================================
with tab2:
    st.subheader("🧪 Synthetic Patient Simulator")
    st.write("Generates a predictive hypnogram based on clinical sleep architecture rules.")
    
    col1, col2 = st.columns(2)
    name_in = col1.text_input("Name:", "Patient_001")
    age_in = col1.slider("Age:", 1, 100, 35)
    hours_in = col2.slider("Sleep Hrs:", 4.0, 12.0, 8.0, 0.5)
    bedtime_in = col2.text_input("Bedtime (HH:MM):", "23:00")
    
    if st.button("🚀 Run Simulation", type="primary"):
        try:
            start_dt = datetime.strptime(bedtime_in, "%H:%M")
            start_dt = start_dt.replace(year=datetime.today().year, month=datetime.today().month, day=datetime.today().day)
        except:
            st.error("Error: Invalid Time Format. Use HH:MM")
            st.stop()

        total_epochs = int(hours_in * 120)
        y_syn_base = []

        for i in range(total_epochs):
            progress = i / total_epochs
            cycle_pos = (i % 180) / 180
            
            if progress < 0.05: stage = 0 
            elif progress < 0.4 and cycle_pos < 0.4:
                stage = 3 if age_in < 60 else 2 
            elif progress > 0.6 and cycle_pos > 0.7: stage = 4 
            else: stage = 2 
            
            y_syn_base.append(stage)

        y_syn_smooth = medfilt(y_syn_base, kernel_size=7).astype(int)

        st.markdown("---")
        fig2, ax2 = plt.subplots(figsize=(15, 5), facecolor='#F5F5F7')
        ax2.set_facecolor('white')
        
        config_sim = {4: (0.0, 0.2, '#FF69B4', 'REM'), 3: (0.2, 0.2, '#6C5CE7', 'N3'),
                      2: (0.4, 0.2, '#A29BFE', 'N2'), 1: (0.6, 0.2, '#D1D1FF', 'N1'), 0: (0.8, 0.2, '#FFB347', 'Awake')}

        for st_id, (y_min, h, col, n) in config_sim.items():
            is_st = (y_syn_smooth == st_id).astype(int)
            diff = np.diff(np.concatenate([[0], is_st, [0]]))
            starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                dt_s = start_dt + timedelta(seconds=int(s*30))
                dt_e = start_dt + timedelta(seconds=int(e*30))
                ax2.broken_barh([(mdates.date2num(dt_s), mdates.date2num(dt_e)-mdates.date2num(dt_s))], (y_min, h), facecolors=col, edgecolors='none')

        ax2.set_ylim(0, 1)
        ax2.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax2.set_yticklabels(['REM', 'N3', 'N2', 'N1', 'Awake'], fontweight='bold')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        for spine in ax2.spines.values(): spine.set_visible(False)
        plt.title(f"HYPOTHETICAL SUBJECT: {name_in} (Age: {age_in}) - Multimodal Analysis", fontsize=16, fontweight='bold')
        st.pyplot(fig2)
        
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", bbox_inches='tight', dpi=300)
        safe_name = name_in.replace(" ", "_")
        st.download_button(
            label="💾 Download Synthetic Hypnogram",
            data=buf2.getvalue(),
            file_name=f"Hypnogram_Sim_{safe_name}.png",
            mime="image/png"
        )