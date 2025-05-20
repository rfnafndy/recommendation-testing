import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load data dan preprocess
destinasiData = pd.read_csv('destinasi.csv')
destinasiData.drop(columns=["Place_Id", "Time_Minutes", "Unnamed: 11", "Unnamed: 12", "Coordinate", "Lat", "Long"], inplace=True)
destinasiData.reset_index(drop=True, inplace=True)

# Feature Engineering
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(destinasiData[['Category', 'City']])
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(destinasiData[['Price', 'Rating']])
combined_features = np.hstack((encoded_features, scaled_features))
similarity_matrix = cosine_similarity(combined_features)

def get_recommendations_by_index(idx, top_n=5):
    sim_scores_raw = similarity_matrix[idx]
    sim_scores = list(enumerate(sim_scores_raw))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended_indices = [i[0] for i in sim_scores]
    return destinasiData.iloc[recommended_indices][['Place_Name', 'Category', 'City', 'Price', 'Rating', 'Description']]

st.title("Sistem Rekomendasi Tempat Wisata")

# Input filter user
selected_city = st.selectbox("Pilih Kota", options=sorted(destinasiData['City'].unique()))
selected_category = st.selectbox("Pilih Kategori", options=sorted(destinasiData['Category'].unique()))
max_price = st.number_input("Harga Maksimal (Rp)", min_value=0, value=int(destinasiData['Price'].max()))
min_rating = st.slider("Rating Minimum", 
                       min_value=float(destinasiData['Rating'].min()), 
                       max_value=float(destinasiData['Rating'].max()), 
                       value=float(destinasiData['Rating'].min()))

if st.button("Cari Rekomendasi"):
    filtered_df = destinasiData[
        (destinasiData['City'] == selected_city) &
        (destinasiData['Category'] == selected_category) &
        (destinasiData['Price'] <= max_price) &
        (destinasiData['Rating'] >= min_rating)
    ]

    if filtered_df.empty:
        st.warning("Tidak ada tempat wisata yang sesuai dengan filter.")
    else:
        # Ambil 5 tempat teratas berdasarkan rating
        top5 = filtered_df.sort_values(by='Rating', ascending=False).head(5)

        st.subheader(f"5 Tempat Wisata Teratas di {selected_city} dengan kategori {selected_category}:")
        for idx, row in top5.iterrows():
            st.markdown(f"### {row['Place_Name']} (Rating: {row['Rating']}, Harga: Rp{row['Price']})")
            st.write(row['Description'])

            # Rekomendasi mirip berdasarkan index tempat ini
            recommendations = get_recommendations_by_index(idx, top_n=3)
            if not recommendations.empty:
                st.markdown("**Rekomendasi Tempat Wisata Mirip:**")
                for _, rec in recommendations.iterrows():
                    st.write(f"- {rec['Place_Name']} ({rec['City']}) - Rating: {rec['Rating']}, Harga: Rp{rec['Price']}")
            st.write("---")
