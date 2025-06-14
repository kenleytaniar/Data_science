import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model, scaler, dan label encoder
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label.pkl")  # perhatikan: label_encoders (jamak)

st.title("Prediksi Obesitas Menggunakan Machine Learning")
st.markdown("Masukkan data untuk memprediksi kategori obesitas.")

# Form input
st.header("Form Input Data Pengguna")

# Input numerik
age = st.number_input("Age", min_value=1.0, step=1.0)
height = st.number_input("Height (meter)", min_value=0.5, step=0.01)
weight = st.number_input("Weight (kg)", min_value=1.0, step=0.1)
fcvc = st.slider("Frekuensi konsumsi sayur (FCVC)", 1.0, 3.0, 2.0)
ncp = st.slider("Jumlah makanan utama (NCP)", 1.0, 4.0, 3.0)
ch2o = st.slider("Konsumsi air harian (CH2O)", 1.0, 3.0, 2.0)
faf = st.slider("Frekuensi aktivitas fisik (FAF)", 0.0, 3.0, 1.0)
tue = st.slider("Waktu penggunaan teknologi (TUE)", 0.0, 2.0, 1.0)

# Input kategorik
gender = st.selectbox("Gender", ['Male', 'Female'])
calc = st.selectbox("Konsumsi alkohol (CALC)", ['no', 'Sometimes', 'Frequently', 'Always'])
favc = st.selectbox("Konsumsi makanan berkadar kalori tinggi (FAVC)", ['yes', 'no'])
scc = st.selectbox("Menghitung kalori (SCC)", ['yes', 'no'])
smoke = st.selectbox("Merokok (SMOKE)", ['yes', 'no'])
history = st.selectbox("Riwayat keluarga overweight", ['yes', 'no'])
caec = st.selectbox("Konsumsi makanan di luar (CAEC)", ['no', 'Sometimes', 'Frequently', 'Always'])
mtrans = st.selectbox("Transportasi (MTRANS)", ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

# Buat dataframe dari input
input_df = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'Height': height,
    'Weight': weight,
    'CALC': calc,
    'FAVC': favc,
    'FCVC': fcvc,
    'NCP': ncp,
    'SCC': scc,
    'SMOKE': smoke,
    'CH2O': ch2o,
    'family_history_with_overweight': history,
    'FAF': faf,
    'TUE': tue,
    'CAEC': caec,
    'MTRANS': mtrans
}])

# Label encoding untuk kolom kategorik
for col in label_encoders:
    if col in input_df.columns:
        val = input_df.at[0, col]
        input_df.at[0, col] = label_encoders[col].transform([val])[0]

# Normalisasi numerik
scaled_input = scaler.transform(input_df)

# Prediksi
if st.button("Prediksi"):
    pred = model.predict(scaled_input)[0]
    st.subheader("Hasil Prediksi:")
    st.success(f"Kategori Obesitas: **{pred}**")
