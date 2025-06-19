import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model, scaler, dan label encoder
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label.pkl")  # Dictionary: kolom -> LabelEncoder

st.title("Prediksi Obesitas Menggunakan Machine Learning")
st.markdown("Masukkan data untuk memprediksi kategori obesitas.")

# Input form
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
# Pilihan inputan tetap teks agar mudah dipilih user
gender = st.selectbox("Gender", ['Male', 'Female'])
calc = st.selectbox("Konsumsi alkohol (CALC)", ['no', 'Sometimes', 'Frequently', 'Always'])
favc = st.selectbox("Konsumsi makanan berkadar kalori tinggi (FAVC)", ['yes', 'no'])
scc = st.selectbox("Menghitung kalori (SCC)", ['yes', 'no'])
smoke = st.selectbox("Merokok (SMOKE)", ['yes', 'no'])
history = st.selectbox("Riwayat keluarga overweight", ['yes', 'no'])
caec = st.selectbox("Konsumsi makanan di luar (CAEC)", ['no', 'Sometimes', 'Frequently', 'Always'])
mtrans = st.selectbox("Transportasi (MTRANS)", ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

# Mapping ke nilai numerik
gender = 1 if gender == 'Male' else 0
favc = 1 if favc == 'yes' else 0
scc = 1 if scc == 'yes' else 0
smoke = 1 if smoke == 'yes' else 0
history = 1 if history == 'yes' else 0

calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
mtrans_map = {"Public_Transportation": 0, "Walking": 1, "Automobile": 2, "Motorbike": 3, "Bike": 4}

calc = calc_map[calc]
caec = caec_map[caec]
mtrans = mtrans_map[mtrans]

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

# Tampilkan input untuk debugging (opsional)
st.write("Input Data:", input_df)

# Label encoding untuk kolom kategorik dengan validasi aman
for col in label_encoders:
    if col in input_df.columns:
        try:
            val = str(input_df.at[0, col])  # pastikan bertipe string
            encoded_val = label_encoders[col].transform([val])[0]
            input_df.at[0, col] = encoded_val
        except Exception as e:
            st.error(f"❌ Nilai '{val}' tidak dikenali untuk kolom '{col}'. Pilih salah satu dari: {label_encoders[col].classes_}")
            st.stop()
# Normalisasi numerik
scaled_input = scaler.transform(input_df)

# Tombol prediksi
if st.button("Prediksi"):
    pred = model.predict(scaled_input)[0]
    st.subheader("Hasil Prediksi:")
    st.success(f"Kategori Obesitas: **{pred}**")
