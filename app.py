import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('destinasi.csv')
    # Drop kolom tidak perlu
    df.drop(columns=["Place_Id", "Time_Minutes", "Unnamed: 11", "Unnamed: 12", "Coordinate", "Lat", "Long"], inplace=True)
    return df

destinasiData = load_data()

# --- Preprocessing fitur ---
@st.cache_data
def preprocess_features(df):
    # One-hot encode Category & City
    encoder = OneHotEncoder(sparse=False)
    encoded_cat_city = encoder.fit_transform(df[['Category', 'City']])
    
    # Normalisasi Price dan Rating
    scaler = MinMaxScaler()
    scaled_price_rating = scaler.fit_transform(df[['Price', 'Rating']])
    
    # Gabungkan semua fitur
    combined_features = np.hstack((encoded_cat_city, scaled_price_rating))
    return combined_features, encoder, scaler

features_matrix, encoder, scaler = preprocess_features(destinasiData)

# Hitung similarity matrix (cosine similarity)
@st.cache_data
def calculate_similarity(matrix):
    return cosine_similarity(matrix)

similarity_matrix = calculate_similarity(features_matrix)

# Fungsi rekomendasi
def get_recommendations(place_name, top_n=5):
    place_name_lower = place_name.lower()
    mask = destinasiData['Place_Name'].str.lower() == place_name_lower
    if not mask.any():
        return None
    idx = destinasiData[mask].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    indices = [i[0] for i in sim_scores]
    return destinasiData.iloc[indices][['Place_Name', 'Category', 'City', 'Price', 'Rating', 'Description']]

# --- Streamlit UI ---
st.title("Sistem Rekomendasi Tempat Wisata (Content-Based Filtering)")

# Filter input user
selected_city = st.selectbox("Pilih Kota", options=sorted(destinasiData['City'].unique()))
selected_category = st.selectbox("Pilih Kategori", options=sorted(destinasiData['Category'].unique()))
max_price = st.number_input("Harga Maksimal (Rp)", min_value=0, value=int(destinasiData['Price'].max()))
min_rating = st.slider("Rating Minimum", min_value=float(destinasiData['Rating'].min()), max_value=float(destinasiData['Rating'].max()), value=float(destinasiData['Rating'].min()))

# Filter data berdasarkan input
filtered_df = destinasiData[
    (destinasiData['City'] == selected_city) &
    (destinasiData['Category'] == selected_category) &
    (destinasiData['Price'] <= max_price) &
    (destinasiData['Rating'] >= min_rating)
]

if filtered_df.empty:
    st.warning("Tidak ada tempat wisata yang sesuai dengan filter.")
else:
    # Tampilkan list nama tempat wisata untuk dipilih
    place_selected = st.selectbox("Pilih Tempat Wisata", options=filtered_df['Place_Name'].values)
    
    if st.button("Cari Rekomendasi Mirip"):
        recommendations = get_recommendations(place_selected, top_n=5)
        if recommendations is None or recommendations.empty:
            st.info("Tempat wisata tidak ditemukan dalam dataset.")
        else:
            st.subheader(f"Rekomendasi Tempat Wisata Mirip dengan '{place_selected}':")
            for idx, row in recommendations.iterrows():
                with st.expander(f"{row['Place_Name']} - {row['City']} (Rating: {row['Rating']}, Harga: Rp{row['Price']})"):
                    st.write(row['Description'])
