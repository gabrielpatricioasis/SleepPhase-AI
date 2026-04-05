# 🌙 SleepPhase-AI: Automated Scoring System

**SleepPhase AI** is an advanced sleep stage analysis platform that integrates physiological signal processing with Artificial Intelligence (MLP) to provide accurate, automated hypnograms.

---

## 🚀 Key Features

* **Real Patient Analysis (PSG):** Seamless processing of `.edf` files using EEG (Fpz-Cz), EOG, and EMG signals.
* **AI-Powered Classification:** A Multilayer Perceptron (MLP) model calibrated against the *Sleep-EDF (Kemp et al.)* database, achieving a verified **92.0%** accuracy.
* **Synthetic Simulator:** Generation of predictive hypnograms based on clinical sleep architecture rules and demographic variables (age, sleep duration, bedtime).
* **Clinical Export:** Built-in function to download generated hypnograms in high-resolution (PNG) format for reporting.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Web Framework:** Streamlit
* **Signal Processing:** MNE-Python, SciPy
* **Machine Learning:** Scikit-learn (MLP Classifier), Joblib
* **Visualization:** Matplotlib

---

## 📂 Project Structure

* `app.py`: Core application logic and user interface.
* `sleep_mlp_model.joblib`: Trained AI model weights.
* `sleep_scaler.joblib`: Data scaler for signal normalization.
* `subjects/`: Directory for storing real patient PSG (.edf) files.
* `requirements.txt`: Necessary dependencies for cloud deployment.

---

## 📋 Analysis Methodology

The system extracts key features from raw physiological signals, including:
1.  **Frequency Bands (EEG):** Delta, Theta, Alpha, Sigma, and Beta.
2.  **Ocular Activity (EOG):** Essential for REM phase detection.
3.  **Muscle Tone (EMG):** Used to differentiate between Wakefulness and Deep Sleep (N3/N4) stages.

---

## 👥 Authors

This project was developed as part of the **Master's in Cognitive Systems and Interactive Media (UPF)**:

* **Gabriel Asís-Sagrado**
* **Èric Domingo Roca**
* **Michaela Freire Griffith**

---
*Disclaimer: This software is intended for academic and research purposes only. For clinical diagnosis, consultation with a board-certified somnologist is required.*
