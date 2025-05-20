# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('destinasi.csv')
    # Drop kolom yang tidak diperlukan
    df = df.drop(columns=["Place_Id", "Time_Minutes", "Unnamed: 11", "Unnamed: 12", "Coordinate", "Lat", "Long"])
    return df

destinasiData = load_data()

# --- Preprocessing fitur dan buat similarity matrix ---
@st.cache_data
def preprocess_features(df):
    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(df[['Category', 'City']])
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Price', 'Rating']])
    
    # Debug print (opsional, bisa dihapus setelah yakin)
    # st.write(f"Encoded shape: {encoded.shape}, type: {type(encoded)}")
    # st.write(f"Scaled shape: {scaled.shape}, type: {type(scaled)}")
    
    combined = np.hstack((encoded, scaled))
    return combined

combined_features = preprocess_features(destinasiData)

@st.cache_data
def compute_similarity_matrix(features):
    return cosine_similarity(features)

similarity_matrix = compute_similarity_matrix(combined_features)

# --- Fungsi rekomendasi ---
def get_recommendations(place_name, top_n=5):
    place_name_lower = place_name.lower()
    place_names_lower = destinasiData['Place_Name'].str.lower()
    
    if place_name_lower not in place_names_lower.values:
        return pd.DataFrame()  # kosongkan hasil jika tidak ditemukan
    
    index = place_names_lower[place_names_lower == place_name_lower].index[0]
    scores = list(enumerate(similarity_matrix[index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended_indices = [i[0] for i in sorted_scores]
    
    return destinasiData.iloc[recommended_indices][['Place_Name', 'Category', 'City', 'Price', 'Rating', 'Description']]

# --- Streamlit UI ---
st.title("Sistem Rekomendasi Tempat Wisata (Content-Based Filtering)")

selected_city = st.selectbox("Pilih Kota", options=sorted(destinasiData['City'].unique()))
selected_category = st.selectbox("Pilih Kategori", options=sorted(destinasiData['Category'].unique()))
max_price = st.number_input("Harga Maksimal (Rp)", min_value=0, value=int(destinasiData['Price'].max()))
min_rating = st.slider("Rating Minimum", min_value=float(destinasiData['Rating'].min()), max_value=float(destinasiData['Rating'].max()), value=float(destinasiData['Rating'].min()))

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
            st.info("Tempat wisata tidak ditemukan dalam dataset.")
        else:
            st.subheader(f"Rekomendasi Tempat Wisata Mirip dengan '{place_selected}':")
            for idx, row in recommendations.iterrows():
                with st.expander(f"{row['Place_Name']} - {row['City']} (Rating: {row['Rating']}, Harga: Rp{row['Price']})"):
                    st.write(row['Description'])
