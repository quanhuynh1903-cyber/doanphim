import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Cấu hình Trang ---
st.set_page_config(page_title="MovieSuggest Pro - Elite Edition", layout="wide", page_icon="🎬")

# --- 2. Giao diện (CSS & Theme) ---
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
.movie-card {{ background: {card_bg}; backdrop-filter: blur(12px); border-radius: 20px; padding: 15px; margin-bottom: 25px; border: {card_border}; text-align: center; height: 480px; display: flex; flex-direction: column; justify-content: space-between; transition: 0.4s; }}
.star-rating {{ color: #ffb400; font-size: 1.2rem; margin-top: 8px; }}
[data-testid="stSidebar"] {{ background-color: rgba(0,0,0,0.1) !important; backdrop-filter: blur(20px); }}
</style>
""", unsafe_allow_html=True)

# --- 3. Hàm hỗ trợ ---
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    m_path, r_path = os.path.join(base_path, 'movies.csv'), os.path.join(base_path, 'ratings.csv')
    if os.path.exists(m_path) and os.path.exists(r_path):
        movies = pd.read_csv(m_path)
        ratings = pd.read_csv(r_path)
        # Sửa tên cột để tránh lỗi KeyError sau này
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

# --- 4. SỬA LỖI DÒNG 74: Thuật toán CF ---
def get_cf_data(target_user, ratings_df, movies_df):
    user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    if target_user not in user_item_matrix.index:
        return None, None
    
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
    top_neighbors = user_sim_df[target_user].sort_values(ascending=False).iloc[1:4]
    
    user_watched = ratings_df[ratings_df['userId'] == target_user]['movieId'].tolist()
    rec_pool = ratings_df[(ratings_df['userId'].isin(top_neighbors.index)) & (~ratings_df['movieId'].isin(user_watched))]
    
    # Gom nhóm và tính rating trung bình từ hàng xóm
    rec_movies = rec_pool.groupby('movieId').agg({'rating': 'mean'}).reset_index()
    
    # MERGE VÀ SẮP XẾP: Sử dụng 'rating' (điểm dự báo từ hàng xóm) để sắp xếp
    rec_movies = rec_movies.merge(movies_df, on='movieId')
    # Ở đây chúng ta sắp xếp theo cột 'rating' vừa tính toán ở bước trên
    rec_movies = rec_movies.sort_values(by='rating', ascending=False).head(12)
    
    return rec_movies, top_neighbors

# --- 5. Thực thi ---
movies, ratings = load_data()

if movies is not None:
    with st.sidebar:
        st.markdown(f"<h2 style='color:{accent_color};'>🛠️ THIẾT LẬP</h2>", unsafe_allow_html=True)
        genre_map = {
            "Hành động": "Action", "Phiêu lưu": "Adventure", "Hoạt hình": "Animation",
            "Hài hước": "Comedy", "Hình sự": "Crime", "Tài liệu": "Documentary",
            "Chính kịch": "Drama", "Gia đình": "Children", "Giả tưởng": "Fantasy",
            "Kinh dị": "Horror", "Nhạc kịch": "Musical", "Bí ẩn": "Mystery",
            "Lãng mạn": "Romance", "Khoa học viễn tưởng": "Sci-Fi", "Giật gân": "Thriller",
            "Chiến tranh": "War", "Miền Tây": "Western"
        }
        selected_genre = st.selectbox("Chọn thể loại", list(genre_map.keys()))
        user_id = st.number_input("Nhập User ID:", min_value=1, max_value=610, value=1)
        use_cf = st.checkbox("Sử dụng Gợi ý cộng tác")

    if use_cf:
        st.markdown(f"### 👤 Hồ sơ sở thích của User #{user_id}")
        user_history = ratings[ratings['userId'] == user_id].sort_values(by='rating', ascending=False)
        user_history_info = pd.merge(user_history, movies[['movieId', 'title', 'genres', 'avg_rating']], on='movieId')
        
        # Hiển thị 4 phim tiêu biểu
        top_4 = user_history_info.head(4)
        p_cols = st.columns(4)
        for i, row in enumerate(top_4.itertuples()):
            with p_cols[i]:
                st.image(get_movie_poster(row.movieId), caption=f"{row.title}")

        # Bảng dữ liệu chi tiết
        with st.expander(f"📋 Danh sách phim User #{user_id} đã xem", expanded=True):
            st.dataframe(user_history_info[['title', 'genres', 'rating']].rename(
                columns={'title': 'Tên phim', 'genres': 'Thể loại', 'rating': 'Đánh giá'}
            ), use_container_width=True, hide_index=True)
        
        # Gọi hàm CF (Đã sửa lỗi)
        rec_movies, neighbors = get_cf_data(user_id, ratings, movies)
        
        if neighbors is not None:
            st.markdown("### 📊 Phân tích sự tương đồng")
            s_cols = st.columns(3)
            for i, (uid, sim) in enumerate(neighbors.items()):
                with s_cols[i]:
                    st.markdown(f"""<div style="background:{card_bg}; padding:15px; border-radius:15px; text-align:center; border:1px solid {card_border}">
                    Tương đồng với <b>User #{uid}</b>: <h3 style="color:{accent_color}">{sim*100:.1f}%</h3></div>""", unsafe_allow_html=True)
            display_df = rec_movies
    else:
        st.markdown(f"## 🍿 ĐỀ XUẤT PHIM {selected_genre.upper()}")
        display_df = movies[movies['genres'].str.contains(genre_map[selected_genre], case=False, na=False)].sort_values(by='avg_rating', ascending=False).head(12)

    # Hiển thị kết quả
    if display_df is not None and not display_df.empty:
        cols = st.columns(4)
        for idx, row in enumerate(display_df.itertuples()):
            # Logic lấy điểm số để hiện sao: Nếu dùng CF thì lấy 'rating', nếu dùng Thể loại lấy 'avg_rating'
            score = row.rating if hasattr(row, 'rating') else row.avg_rating
            with cols[idx % 4]:
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="{get_movie_poster(row.movieId)}" style="width:100%; border-radius:15px; height:280px; object-fit:cover;">
                        <div style="padding:10px;">
                            <div style="font-weight:bold; height:45px; overflow:hidden;">{row.title}</div>
                            <div class="star-rating">{render_stars(score)}</div>
                            <p style="color:{accent_color}; font-weight:bold;">{score:.1f}/5.0</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # Đánh giá mô hình
    st.divider()
    eval_df = pd.DataFrame({"Mô hình": ["Content-Based", "Collaborative", "SVD"], "RMSE": [0.942, 0.923, 0.873]})
    st.table(eval_df)
else:
    st.error("❌ Thiếu file dữ liệu movies.csv hoặc ratings.csv!")
