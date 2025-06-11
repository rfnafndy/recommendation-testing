# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- 1. INISIALISASI STEMMER ---
stemmer = StemmerFactory().create_stemmer()

# --- 2. FUNGSI MEMBERSIHKAN TEKS ---
def bersihkan_text(teks):
    teks = str(teks).lower()
    teks = re.sub(r'[^a-z0-9\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return stemmer.stem(teks)

# --- 3. LOAD DATA ---
@st.cache_data
def load_data():
    destinasi = pd.read_csv("destinasi.csv")
    rating = pd.read_csv("rating.csv")

    destinasi.drop(columns=['Coordinate', 'Lat', 'Long', 'Unnamed: 11', 'Unnamed: 12', 'Time_Minutes'],
                   errors='ignore', inplace=True)

    data = pd.merge(rating, destinasi, on='Place_Id', how='left')
    data.drop_duplicates(subset=['Place_Name'], inplace=True)
    data = data.reset_index(drop=True)

    data['Clean_Desc'] = data['Description'].fillna('').apply(bersihkan_text)
    data['Clean_Category'] = data['Category'].fillna('').apply(bersihkan_text)
    data['Features'] = data['Clean_Desc'] + ' ' + data['Clean_Category']

    return data

# --- 4. FUNGSI REKOMENDASI ---
def buat_model(data):
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2)
    tfidf_matrix = tfidf.fit_transform(data['Features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def rekomendasi_wisata(data, cosine_sim, nama_tempat, k=5, lokasi=None, harga_max=None, rating_min=None):
    nama_tempat = nama_tempat.lower()
    matches = data[data['Place_Name'].str.lower() == nama_tempat]

    if matches.empty:
        return f"Tempat wisata '{nama_tempat}' tidak ditemukan."

    idx = matches.index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    rekomendasi_df = pd.DataFrame([
        {
            'Place_Name': data.at[i, 'Place_Name'],
            'Category': data.at[i, 'Category'],
            'Location': data.at[i, 'City'] if 'City' in data.columns else '',
            'Price': data.at[i, 'Price'],
            'Rating': data.at[i, 'Rating'],
            'Similarity': round(score, 3)
        }
        for i, score in similarity_scores[1:]
        if i != idx
    ])

    if lokasi:
        rekomendasi_df = rekomendasi_df[rekomendasi_df['Location'].str.lower() == lokasi.lower()]
    if harga_max is not None:
        rekomendasi_df = rekomendasi_df[rekomendasi_df['Price'].fillna(0) <= harga_max]
    if rating_min is not None:
        rekomendasi_df = rekomendasi_df[rekomendasi_df['Rating'].fillna(0) >= rating_min]

    return rekomendasi_df.head(k) if not rekomendasi_df.empty else "Tidak ada rekomendasi sesuai filter."

# --- 5. UI STREAMLIT ---
st.set_page_config(page_title="Rekomendasi Wisata", layout="wide")
st.title("Sistem Rekomendasi Tempat Wisata")

data = load_data()
cosine_sim = buat_model(data)

# Input pengguna
nama_tempat = st.text_input("Nama Tempat yang Pernah Dikunjungi", placeholder="Misal: Kota Tua")
k = st.slider("Jumlah Rekomendasi", 1, 10, 5)
lokasi = st.text_input("Filter Lokasi (Opsional)", placeholder="Misal: Jakarta")
harga_max = st.number_input("Harga Maksimum (Opsional)", min_value=0, value=100000)
rating_min = st.slider("Rating Minimum (Opsional)", 0.0, 5.0, 4.0, 0.1)

if st.button("Tampilkan Rekomendasi"):
    if nama_tempat.strip() == "":
        st.warning("Mohon isi nama tempat terlebih dahulu.")
    else:
        hasil = rekomendasi_wisata(data, cosine_sim, nama_tempat, k, lokasi, harga_max, rating_min)
        if isinstance(hasil, pd.DataFrame):
            st.success(f"Hasil Rekomendasi Mirip '{nama_tempat.title()}':")
            st.dataframe(hasil, use_container_width=True)
        else:
            st.error(hasil)
