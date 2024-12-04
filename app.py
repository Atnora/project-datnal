import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Judul aplikasi
st.title('Prediksi Dampak Mental Berdasarkan Penggunaan Teknologi')

st.write("""
    Aplikasi ini menggunakan model Support Vector Machine (SVM) untuk memprediksi dampak mental seseorang berdasarkan
    data penggunaan teknologi, media sosial, gaming, tingkat stres, jam tidur, dan dampak lingkungan kerja.
    Silakan isi data yang diminta dan aplikasi ini akan memberikan hasil prediksi dan probabilitasnya.
""")

# Fungsi untuk memuat dataset dan memprosesnya
def load_data():
    dataset = pd.read_csv('mental_health_and_technology_usage_2024.csv')
    return dataset

# Menampilkan dataset untuk verifikasi
st.write("Dataset yang dimuat:")
data = load_data()
st.write(data.head())

# Input data dari pengguna
technology_hours = st.number_input("Masukkan jam penggunaan teknologi (integer)", min_value=0, max_value=24)
social_media_hours = st.number_input("Masukkan jam penggunaan media sosial (integer)", min_value=0, max_value=24)
gaming_hours = st.number_input("Masukkan jam bermain game (integer)", min_value=0, max_value=24)
stress_level = st.selectbox("Pilih tingkat stres", ["Low", "Medium", "High"])
sleep_hours = st.number_input("Masukkan jam tidur (integer)", min_value=0, max_value=24)
environmental_impact = st.selectbox("Pilih dampak lingkungan kerja", ["Positive", "Negative"])

# Menyusun data input pengguna
input_data = {
    "Technology_Usage_Hours": technology_hours,
    "Social_Media_Usage_Hours": social_media_hours,
    "Gaming_Hours": gaming_hours,
    "Stress_Level": stress_level,
    "Sleep_Hours": sleep_hours,
    "Work_Environment_Impact": environmental_impact
}

st.write("Data yang Anda masukkan:")
st.write(input_data)

# Preprocessing data
df = data.copy()

# Menggunakan LabelEncoder untuk variabel kategorikal
label_encoder = LabelEncoder()

# Melakukan encoding pada kolom kategorikal
df['Stress_Level'] = label_encoder.fit_transform(df['Stress_Level'])
df['Work_Environment_Impact'] = label_encoder.fit_transform(df['Work_Environment_Impact'])
df['Mental_Health_Status'] = label_encoder.fit_transform(df['Mental_Health_Status'])

# Menyiapkan data fitur dan target
X = df[['Technology_Usage_Hours', 'Social_Media_Usage_Hours', 'Gaming_Hours',
        'Stress_Level', 'Sleep_Hours', 'Work_Environment_Impact']]
y = df['Mental_Health_Status']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model Support Vector Machine (SVM)
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Melakukan prediksi untuk data uji
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

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

st.write(f"Prediksi dampak mental: {predicted_class[0]}")
st.write(f"Probabilitas masing-masing kelas: {prediction_prob[0]}")
