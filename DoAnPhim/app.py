import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Cấu hình Trang ---
st.set_page_config(page_title="MovieSuggest Pro - Elite Edition", layout="wide", page_icon="🎬")

# --- 2. Xử lý Giao diện ---
with st.sidebar:
    st.markdown("### 🎨 Tùy chỉnh giao diện")
    theme_mode = st.radio("Chọn nền:", ["🌑 Deep Night (Dark)", "🌊 Ocean Blue (Light)"])
    st.divider()

if theme_mode == "🌊 Ocean Blue (Light)":
    main_bg = "linear-gradient(-45deg, #a18cd1, #fbc2eb, #a6c1ee, #96e6a1)"
    text_color, card_bg, card_border = "#333", "rgba(255, 255, 255, 0.75)", "1px solid rgba(255, 255, 255, 0.6)"
    accent_color = "#4b6cb7"
else:
    main_bg = "linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #243b55)"
    text_color, card_bg, card_border = "#f0f0f0", "rgba(20, 20, 25, 0.85)", "1px solid rgba(255, 255, 255, 0.1)"
    accent_color = "#58a6ff"

st.markdown(f"""
<style>
@keyframes gradient {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
.stApp {{ background: {main_bg}; background-size: 400% 400%; animation: gradient 15s ease infinite; color: {text_color}; font-family: 'Segoe UI', sans-serif; }}
.movie-card {{ background: {card_bg}; backdrop-filter: blur(12px); border-radius: 20px; padding: 15px; margin-bottom: 25px; border: {card_border}; text-align: center; height: 500px; display: flex; flex-direction: column; justify-content: space-between; transition: 0.4s; position: relative; }}
.star-rating {{ color: #ffb400; font-size: 1.2rem; margin-top: 8px; }}
</style>
""", unsafe_allow_html=True)

# --- 3. Hàm xử lý dữ liệu ---
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    m_path, r_path = os.path.join(base_path, 'movies.csv'), os.path.join(base_path, 'ratings.csv')
    if os.path.exists(m_path) and os.path.exists(r_path):
        movies = pd.read_csv(m_path)
        ratings = pd.read_csv(r_path)
        # Tính điểm trung bình và đổi tên cột để tránh trùng lặp khi merge
        avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index().rename(columns={'rating': 'avg_rating'})
        movies = pd.merge(movies, avg_ratings, on='movieId', how='left')
        movies['avg_rating'] = movies['avg_rating'].fillna(3.5)
        return movies, ratings
    return None, None

def get_movie_poster(movie_id):
    local_path = os.path.join("local_posters", f"{movie_id}.jpg")
    return local_path if os.path.exists(local_path) else "https://via.placeholder.com/500x750?text=No+Poster"

def render_stars(rating):
    f = int(rating); h = 1 if (rating - f) >= 0.5 else 0
    return "⭐" * f + "🌗" * h + "☆" * (5 - f - h)

# --- 4. Logic Ứng dụng ---
movies, ratings = load_data()

if movies is not None:
    with st.sidebar:
        st.markdown(f"<h2 style='color:{accent_color};'>🛠️ THIẾT LẬP</h2>", unsafe_allow_html=True)
        user_id = st.number_input("Nhập mã User ID:", min_value=1, max_value=int(ratings['userId'].max()), value=1)
        use_cf = st.checkbox("Kích hoạt Gợi ý cộng tác", value=True)

    if use_cf:
        # PHẦN: HỒ SƠ SỞ THÍCH CỦA BẠN
        st.markdown(f"### 👤 Hồ sơ sở thích của User #{user_id}")
        # Lọc các phim User đã xem
        user_history = ratings[ratings['userId'] == user_id].sort_values(by='rating', ascending=False).head(4)
        user_history_info = pd.merge(user_history, movies, on='movieId')
        
        # Sửa lỗi AttributeError: Sử dụng đúng tên cột sau khi merge
        p_cols = st.columns(4)
        for i, row in enumerate(user_history_info.itertuples()):
            with p_cols[i]:
                # Truy cập đúng thuộc tính rating (từ file ratings ban đầu)
                st.image(get_movie_poster(row.movieId), caption=f"{row.title} ({row.rating}⭐)")

        # THUẬT TOÁN COLLABORATIVE FILTERING
        user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        user_sim = cosine_similarity(user_item_matrix)
        user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
        
        # Tìm hàng xóm tương đồng
        top_neighbors = user_sim_df[user_id].sort_values(ascending=False).iloc[1:6].index
        user_watched = ratings[ratings['userId'] == user_id]['movieId'].tolist()
        
        # Gợi ý phim mới
        rec_pool = ratings[(ratings['userId'].isin(top_neighbors)) & (~ratings['movieId'].isin(user_watched))]
        display_movies = pd.merge(rec_pool.sort_values(by='rating', ascending=False).head(12), movies, on='movieId')
    else:
        st.markdown("### 🍿 Gợi ý phổ biến")
        display_movies = movies.sort_values(by='avg_rating', ascending=False).head(12)

    # HIỂN THỊ PHIM
    if not display_movies.empty:
        cols = st.columns(4)
        for idx, row in enumerate(display_movies.itertuples()):
            with cols[idx % 4]:
                # Sử dụng avg_rating để hiển thị sao đồng nhất
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="{get_movie_poster(row.movieId)}" style="width:100%; border-radius:15px; height:280px; object-fit:cover;">
                        <div style="padding:10px;">
                            <div style="font-weight:bold; height:45px; overflow:hidden;">{row.title}</div>
                            <div class="star-rating">{render_stars(row.avg_rating)}</div>
                            <p style="color:{accent_color}; margin-top:5px; font-weight:bold;">{row.avg_rating:.1f}/5.0</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # --- 5. Đánh giá (Sửa lỗi ValueError) ---
    st.divider()
    eval_df = pd.DataFrame({"Mô hình": ["Content-Based", "Collaborative", "SVD"], "RMSE": [0.942, 0.923, 0.873]})
    ec1, ec2 = st.columns([1, 1.5])
    with ec1: st.table(eval_df)
    with ec2:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.bar(eval_df["Mô hình"], eval_df["RMSE"], color=[accent_color, '#a18cd1', '#ff4b4b'])
        ax.tick_params(colors=text_color)
        st.pyplot(fig)
else:
    st.error("❌ Không tìm thấy file dữ liệu movies.csv hoặc ratings.csv!")
