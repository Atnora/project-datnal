import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Judul aplikasi
st.title('Prediksi Dampak Mental Berdasarkan Penggunaan Teknologi')

# Deskripsi aplikasi
st.write("""
    Aplikasi ini menggunakan model Support Vector Machine (SVM) untuk memprediksi dampak mental seseorang berdasarkan
    data penggunaan teknologi, media sosial, gaming, tingkat stres, jam tidur, dan dampak lingkungan kerja.
    Silakan isi data yang diminta dan aplikasi ini akan memberikan hasil prediksi dan probabilitasnya.
""")

# Fungsi untuk memuat dataset dan memprosesnya
def load_data():
    # Misalnya dataset yang diunggah adalah CSV
    dataset = pd.read_csv('mental_health_and_technology_usage_2024.csv')  # Ganti dengan path dataset yang sesuai
    return dataset

# Menampilkan dataset untuk verifikasi
st.write("Dataset yang dimuat:")
data = load_data()
st.write(data.head())  # Menampilkan beberapa baris pertama untuk verifikasi

# Input data dari pengguna
technology_hours = st.number_input("Masukkan jam penggunaan teknologi (integer)", min_value=0, max_value=24)
social_media_hours = st.number_input("Masukkan jam penggunaan media sosial (integer)", min_value=0, max_value=24)
gaming_hours = st.number_input("Masukkan jam bermain game (integer)", min_value=0, max_value=24)
stress_level = st.selectbox("Pilih tingkat stres", ["Low", "Medium", "High"])
sleep_hours = st.number_input("Masukkan jam tidur (integer)", min_value=0, max_value=24)
environmental_impact = st.selectbox("Pilih dampak lingkungan kerja", ["Positive", "Negative"])

# Menyusun data input pengguna
input_data = {
    "Technology Usage (hours)": technology_hours,
    "Social Media Usage (hours)": social_media_hours,
    "Gaming Hours": gaming_hours,
    "Stress Level": stress_level,
    "Sleep Hours": sleep_hours,
    "Environmental Impact": environmental_impact
}

# Menampilkan data yang dimasukkan
st.write("Data yang Anda masukkan:")
st.write(input_data)

# Preprocessing data
# Gantilah bagian ini dengan kolom yang sesuai di dataset Anda.
# Jika data Anda memiliki kolom seperti 'stress_level', 'technology_usage', dsb, ubah nama kolom berikut
df = data.copy()

# Menggunakan LabelEncoder untuk variabel kategorikal
label_encoder = LabelEncoder()

# Melakukan encoding pada kolom kategorikal (ubah sesuai nama kolom yang tepat di dataset Anda)
df['Stress Level'] = label_encoder.fit_transform(df['Stress Level'])
df['Environmental Impact'] = label_encoder.fit_transform(df['Environmental Impact'])
df['Mental Health Impact'] = label_encoder.fit_transform(df['Mental Health Impact'])

# Menyiapkan data fitur dan target
X = df.drop("Mental Health Impact", axis=1)
y = df["Mental Health Impact"]

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model Support Vector Machine (SVM)
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Melakukan prediksi untuk data uji
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Menampilkan akurasi model
st.write(f"Akurasi model SVM: {accuracy:.2f}")

# Mengonversi input pengguna ke format yang sama dengan data latih
input_data_transformed = np.array([
    technology_hours,
    social_media_hours,
    gaming_hours,
    label_encoder.transform([stress_level])[0],
    sleep_hours,
    label_encoder.transform([environmental_impact])[0]
]).reshape(1, -1)

# Melakukan prediksi dengan model SVM
prediction_prob = svm_model.predict_proba(input_data_transformed)
predicted_class = label_encoder.inverse_transform([np.argmax(prediction_prob)])

# Menampilkan hasil prediksi
st.write(f"Prediksi dampak mental: {predicted_class[0]}")
st.write(f"Probabilitas masing-masing kelas: {prediction_prob[0]}")

