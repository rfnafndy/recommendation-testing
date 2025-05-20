# -*- coding: utf-8 -*-
"""Informatika Pariwisata"""

# Import library
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Membaca dataset dari file CSV
destinasiData = pd.read_csv('destinasi.csv')

# Drop kolom yang tidak diperlukan
destinasiData.drop(columns=["Place_Id", "Time_Minutes", "Unnamed: 11", "Unnamed: 12", "Coordinate", "Lat", "Long"], inplace=True)

# Reset index agar konsisten
destinasiData.reset_index(drop=True, inplace=True)

# ============================
# Feature Engineering
# ============================

# One-hot encoding untuk 'Category' dan 'City'
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(destinasiData[['Category', 'City']])

# Normalisasi kolom 'Price' dan 'Rating'
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(destinasiData[['Price', 'Rating']])

# Gabungkan fitur encoded dan scaled
combined_features = np.hstack((encoded_features, scaled_features))

# Membuat similarity matrix
similarity_matrix = cosine_similarity(combined_features)

# ============================
# Fungsi Rekomendasi
# ============================

def get_recommendations(place_name, top_n=5):
    place_name_lower = place_name.lower()
    # Cek apakah tempat ada di dataset
    mask = destinasiData['Place_Name'].str.lower() == place_name_lower
    if not mask.any():
        return pd.DataFrame()  # Kosongkan jika tidak ada
    
    idx = destinasiData[mask].index[0]

    # Ambil similarity scores baris index tsb
    sim_scores_raw = similarity_matrix[idx]

    # Buat list (index, score)
    sim_scores = list(enumerate(sim_scores_raw))

    # Urutkan berdasarkan skor similarity tertinggi kecuali indeks itu sendiri
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    # Ambil indeks dari tempat rekomendasi
    recommended_indices = [i[0] for i in sim_scores]

    # Return dataframe dengan kolom yang relevan
    return destinasiData.iloc[recommended_indices][['Place_Name', 'Category', 'City', 'Price', 'Rating', 'Description']]

# ============================
# Streamlit UI
# ============================

st.title("Sistem Rekomendasi Tempat Wisata (Content-Based Filtering)")

# Filter input user
selected_city = st.selectbox("Pilih Kota", options=sorted(destinasiData['City'].unique()))
selected_category = st.selectbox("Pilih Kategori", options=sorted(destinasiData['Category'].unique()))
max_price = st.number_input("Harga Maksimal (Rp)", min_value=0, value=int(destinasiData['Price'].max()))
min_rating = st.slider("Rating Minimum", 
                       min_value=float(destinasiData['Rating'].min()), 
                       max_value=float(destinasiData['Rating'].max()), 
                       value=float(destinasiData['Rating'].min()))

# Filter data sesuai pilihan user
filtered_df = destinasiData[
    (destinasiData['City'] == selected_city) &
    (destinasiData['Category'] == selected_category) &
    (destinasiData['Price'] <= max_price) &
    (destinasiData['Rating'] >= min_rating)
]

if filtered_df.empty:
    st.warning("Tidak ada tempat wisata yang sesuai dengan filter.")
else:
    place_selected = st.selectbox("Pilih Tempat Wisata", options=filtered_df['Place_Name'].values)

    if st.button("Cari Rekomendasi Mirip"):
        recommendations = get_recommendations(place_selected, top_n=5)
        if recommendations.empty:
            st.info("Tempat wisata tidak ditemukan dalam dataset atau tidak ada rekomendasi.")
        else:
            st.subheader(f"Rekomendasi Tempat Wisata Mirip dengan '{place_selected}':")
            for idx, row in recommendations.iterrows():
                with st.expander(f"{row['Place_Name']} - {row['City']} (Rating: {row['Rating']}, Harga: Rp{row['Price']})"):
                    st.write(row['Description'])
