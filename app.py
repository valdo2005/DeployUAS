import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ========================================================================================
# Konfigurasi Halaman dan Judul Aplikasi
# ========================================================================================
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('Aplikasi Prediksi Penyakit Jantung')
st.write("""
Aplikasi ini menggunakan model Machine Learning untuk memprediksi kemungkinan seseorang menderita penyakit jantung berdasarkan data klinis mereka.
Aplikasi ini adalah bagian dari Proyek Ujian Akhir Semester Mata Kuliah Data Mining.
""")

# ========================================================================================
# Memuat Model dan Scaler yang Telah Disimpan
# ========================================================================================
# Menggunakan cache untuk mempercepat pemuatan model saat aplikasi dijalankan kembali
@st.cache_resource
def load_model():
    try:
        with open('model_penyakit_jantung.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("File model 'model_penyakit_jantung.pkl' tidak ditemukan. Pastikan file berada di folder yang sama dengan app.py")
        return None

@st.cache_resource
def load_scaler():
    try:
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except FileNotFoundError:
        st.error("File scaler 'scaler.pkl' tidak ditemukan. Pastikan file berada di folder yang sama dengan app.py")
        return None

model = load_model()
scaler = load_scaler()

# Kolom numerik dan nama kolom asli (untuk memastikan urutan)
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
original_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# ========================================================================================
# Membuat Antarmuka Pengguna (Sidebar untuk Input)
# ========================================================================================
st.sidebar.header('Input Data Pasien')

# Membuat dictionary untuk mapping input user-friendly ke nilai numerik
sex_map = {'Laki-laki': 1, 'Perempuan': 0}
cp_map = {'Typical Angina': 1, 'Atypical Angina': 2, 'Non-anginal Pain': 3, 'Asymptomatic': 4}
fbs_map = {'Ya (> 120 mg/dl)': 1, 'Tidak (<= 120 mg/dl)': 0}
restecg_map = {'Normal': 0, 'Kelainan Gelombang ST-T': 1, 'Hipertrofi Ventrikel Kiri': 2}
exang_map = {'Ya': 1, 'Tidak': 0}
slope_map = {'Upsloping': 1, 'Flat': 2, 'Downsloping': 3}
thal_map = {'Normal': 3, 'Fixed Defect': 6, 'Reversable Defect': 7}

# Mengambil input dari pengguna menggunakan widget Streamlit
age = st.sidebar.slider('Umur', 20, 80, 50)
sex_label = st.sidebar.selectbox('Jenis Kelamin', list(sex_map.keys()))
cp_label = st.sidebar.selectbox('Tipe Nyeri Dada (Chest Pain Type)', list(cp_map.keys()))
trestbps = st.sidebar.slider('Tekanan Darah Saat Istirahat (mm Hg)', 90, 200, 120)
chol = st.sidebar.slider('Kolesterol Serum (mg/dl)', 120, 570, 240)
fbs_label = st.sidebar.selectbox('Gula Darah Puasa > 120 mg/dl?', list(fbs_map.keys()))
restecg_label = st.sidebar.selectbox('Hasil Elektrokardiogram', list(restecg_map.keys()))
thalach = st.sidebar.slider('Detak Jantung Maksimum', 70, 205, 150)
exang_label = st.sidebar.selectbox('Nyeri Dada Akibat Olahraga?', list(exang_map.keys()))
oldpeak = st.sidebar.slider('Depresi ST Akibat Olahraga', 0.0, 6.2, 1.0, 0.1)
slope_label = st.sidebar.selectbox('Kemiringan Puncak Segmen ST', list(slope_map.keys()))
ca = st.sidebar.slider('Jumlah Pembuluh Darah Utama (diwarnai fluoroskopi)', 0, 3, 0)
thal_label = st.sidebar.selectbox('Kelainan Thalasemia', list(thal_map.keys()))

# Mengonversi input label menjadi nilai numerik
sex = sex_map[sex_label]
cp = cp_map[cp_label]
fbs = fbs_map[fbs_label]
restecg = restecg_map[restecg_label]
exang = exang_map[exang_label]
slope = slope_map[slope_label]
thal = thal_map[thal_label]

# ========================================================================================
# Tombol Prediksi dan Menampilkan Hasil
# ========================================================================================
if st.sidebar.button('Prediksi Penyakit Jantung'):
    if model is None or scaler is None:
        st.error("Tidak dapat melakukan prediksi karena model atau scaler gagal dimuat.")
    else:
        # Membuat dictionary dari input data
        input_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }

        # Mengubah dictionary menjadi DataFrame dengan urutan kolom yang benar
        input_df = pd.DataFrame([input_data])
        input_df = input_df[original_columns]

        # Melakukan scaling pada fitur numerik
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        
        # Melakukan prediksi
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader('Hasil Prediksi')
        
        if prediction[0] == 1:
            st.error('**Pasien diprediksi MEMILIKI PENYAKIT JANTUNG**')
            st.write(f"**Probabilitas:** {prediction_proba[0][1]*100:.2f}%")
        else:
            st.success('**Pasien diprediksi SEHAT (Tidak Memiliki Penyakit Jantung)**')
            st.write(f"**Probabilitas:** {prediction_proba[0][0]*100:.2f}%")

        st.subheader("Detail Input Pasien:")
        # Menampilkan input pengguna dalam bentuk tabel yang lebih rapi
        display_data = {
            "Fitur": list(input_data.keys()),
            "Nilai": list(input_data.values())
        }
        st.table(pd.DataFrame(display_data))
