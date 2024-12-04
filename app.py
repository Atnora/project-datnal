import streamlit as st
import pandas as pd

# Judul aplikasi
st.title("Mental Health and Technology Usage Dashboard")

# Membaca dataset
@st.cache
def load_data():
    # Ganti dengan path dataset kamu jika ada di folder lain
    data = pd.read_csv('D:\UMN\SEMESTER 3\Data Analysis\mental_health_and_technology_usage_2024.csv')
    return data

# Memuat data
data = load_data()

# Menampilkan dataset
st.subheader("Dataset:")
st.write(data.head())

# Deskripsi statistik
st.subheader("Descriptive Statistics:")
st.write(data.describe())

# Pilih kolom untuk ditampilkan
st.subheader("Pilih Kolom:")
columns = data.columns.tolist()
selected_column = st.selectbox("Pilih Kolom untuk Ditampilkan", columns)

st.write(data[selected_column])

# Memungkinkan pengguna untuk memilih filter berdasarkan beberapa kolom (misalnya kategori jenis kelamin)
st.subheader("Filter Berdasarkan Gender:")
gender_filter = st.selectbox("Pilih Gender", data['Gender'].unique())
filtered_data = data[data['Gender'] == gender_filter]
st.write(filtered_data)
