import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- Cấu hình Trang ---
st.set_page_config(page_title="MovieFlix Premium", layout="wide", page_icon="🍿")

# --- CUSTOM CSS: Giao diện Cinema hiện đại ---
st.markdown("""
    <style>
    /* Nền tối chủ đạo */
    .stApp { background-color: #0e1117; }
    
    /* Sidebar nổi bật */
    [data-testid="stSidebar"] {
        background-image: linear-gradient(#2e3440, #0e1117);
        border-right: 1px solid #e50914;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #e50914 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* Thẻ phim (Movie Card) */
    .movie-container {
        background-color: #1a1c23;
        padding: 15px;
        border-radius: 15px;
        border: 1px solid #30363d;
        text-align: center;
        transition: transform .3s;
        height: 400px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .movie-container:hover {
        transform: scale(1.05);
        border-color: #e50914;
        box-shadow: 0 10px 20px rgba(229, 9, 20, 0.2);
    }
    .movie-title {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: bold;
        margin: 10px 0;
        height: 50px;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .movie-genre {
        color: #b3b3b3;
        font-size: 0.85rem;
    }

    /* Nút bấm đỏ đặc trưng */
    .stButton>button {
        background-color: #e50914;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff0a16;
        box-shadow: 0 0 15px #e50914;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Logic tải dữ liệu ---
@st.cache_data
def load_movie_data():
    base_path = os.path.dirname(__file__)
    movies_path = os.path.join(base_path, 'movies.csv')
    ratings_path = os.path.join(base_path, 'ratings.csv')
    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        return None, None
    return pd.read_csv(movies_path), pd.read_csv(ratings_path)

movies, ratings = load_movie_data()

if movies is not None:
    # Xử lý ma trận (Collaborative Filtering)
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    # --- SIDEBAR: Phần đăng nhập ---
    with st.sidebar:
        st.markdown("# 👤 THÀNH VIÊN")
        st.info("Nhập mã ID để nhận gợi ý cá nhân hóa")
        user_id = st.number_input("User ID (1 - 610):", min_value=1, max_value=610, value=1)
        num_rec = st.slider("Số lượng gợi ý:", 4, 12, 8)
        st.divider()
        st.write("Hệ thống: **Collaborative Filtering**")
        st.write("Dữ liệu: **MovieLens 100k**")

    # --- MAIN CONTENT ---
    st.markdown("<h1 style='text-align: center; color: #e50914;'>🍿 MOVIEFLIX PREMIUM</h1>", unsafe_allow_html=True)
    
    # Gợi ý phim
    st.markdown(f"### 🌟 Danh sách đề xuất cho User #{user_id}")
    
    # Logic lấy phim
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:11].index
    similar_ratings = user_item_matrix.loc[similar_users].mean(axis=0)
    user_watched = user_item_matrix.loc[user_id]
    rec_ids = similar_ratings[user_watched == 0].sort_values(ascending=False).head(num_rec).index
    rec_movies = movies[movies['movieId'].isin(rec_ids)]

    # Hiển thị Grid
    cols = st.columns(4) # Chia 4 cột cho đẹp
    for idx, (_, row) in enumerate(rec_movies.iterrows()):
        with cols[idx % 4]:
            st.markdown(f"""
                <div class="movie-container">
                    <img src="https://via.placeholder.com/200x280/1a1c23/ffffff?text=FILM" style="width:100%; border-radius:10px;">
                    <div>
                        <div class="movie-title">{row['title']}</div>
                        <div class="movie-genre">{row['genres'].split('|')[0]}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("▶ Xem ngay", key=f"btn_{row['movieId']}"):
                st.toast(f"Đang tải: {row['title']}")

    # Phim Xu hướng
    st.markdown("---")
    st.markdown("### 🔥 Xu hướng hiện nay")
    trending = movies.sample(4)
    t_cols = st.columns(4)
    for idx, (_, row) in enumerate(trending.iterrows()):
        with t_cols[idx]:
            st.markdown(f"""
                <div class="movie-container">
                    <img src="https://via.placeholder.com/200x280/e50914/ffffff?text=TRENDING" style="width:100%; border-radius:10px;">
                    <div class="movie-title">{row['title']}</div>
                </div>
            """, unsafe_allow_html=True)

else:
    st.error("Thiếu file dữ liệu!")
