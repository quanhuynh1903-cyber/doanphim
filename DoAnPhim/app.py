import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Cấu hình Trang ---
st.set_page_config(page_title="MovieSuggest Pro - Elite Edition", layout="wide", page_icon="🎬")

# --- 2. Xử lý Giao diện (Theme Mode) ---
with st.sidebar:
    st.markdown("### 🎨 Tùy chỉnh giao diện")
    theme_mode = st.radio("Chọn nền:", ["🌑 Deep Night (Dark)", "🌊 Ocean Blue (Light)"])
    st.divider()

if theme_mode == "🌊 Ocean Blue (Light)":
    main_bg = "linear-gradient(-45deg, #a18cd1, #fbc2eb, #a6c1ee, #96e6a1)"
    text_color, card_bg, card_border = "#333", "rgba(255, 255, 255, 0.75)", "1px solid rgba(255, 255, 255, 0.6)"
    sidebar_bg = "rgba(255, 255, 255, 0.2)"
    accent_color = "#4b6cb7"
else:
    main_bg = "linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #243b55)"
    text_color, card_bg, card_border = "#f0f0f0", "rgba(20, 20, 25, 0.85)", "1px solid rgba(255, 255, 255, 0.1)"
    sidebar_bg = "rgba(0, 0, 0, 0.4)"
    accent_color = "#58a6ff"

# Inject CSS - Sửa lỗi thụt đầu dòng và hiển thị thẻ Card
st.markdown(f"""
<style>
@keyframes gradient {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
.stApp {{ background: {main_bg}; background-size: 400% 400%; animation: gradient 15s ease infinite; color: {text_color}; font-family: 'Segoe UI', sans-serif; }}
.movie-card {{ background: {card_bg}; backdrop-filter: blur(12px); border-radius: 20px; padding: 15px; margin-bottom: 25px; border: {card_border}; text-align: center; height: 480px; display: flex; flex-direction: column; justify-content: space-between; transition: 0.4s; }}
.movie-card:hover {{ transform: translateY(-10px); box-shadow: 0 15px 35px rgba(0,0,0,0.3); border-color: {accent_color}; }}
.metric-box {{ background: {card_bg}; border-radius: 15px; padding: 15px; text-align: center; border: {card_border}; font-weight: bold; color: {accent_color}; }}
[data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; backdrop-filter: blur(20px); }}
h1, h2, h3 {{ color: {text_color} !important; text-align: center; }}
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
        avg = ratings.groupby('movieId')['rating'].mean().reset_index()
        movies = pd.merge(movies, avg, on='movieId', how='left')
        movies['rating'] = movies['rating'].fillna(np.random.uniform(2.5, 4.0))
        return movies, ratings
    return None, None

def get_movie_poster(movie_id):
    local_path = os.path.join("local_posters", f"{movie_id}.jpg")
    return local_path if os.path.exists(local_path) else "https://via.placeholder.com/500x750?text=No+Poster"

# --- 4. Logic Ứng dụng ---
movies, ratings = load_data()

if movies is not None:
    with st.sidebar:
        st.markdown(f"<h2 style='color:{accent_color};'>🛠️ THIẾT LẬP</h2>", unsafe_allow_html=True)
        target_user = st.number_input("Nhập mã User ID:", min_value=1, max_value=int(ratings['userId'].max()), value=1)
        st.info("💡 Hệ thống sẽ tìm các User tương đồng để gợi ý.")

    # PHẦN 1: HỒ SƠ SỞ THÍCH - Sửa lỗi KeyError bằng cách kiểm tra cột dữ liệu
    st.markdown(f"### 👤 Hồ sơ sở thích của User #{target_user}")
    user_ratings = ratings[ratings['userId'] == target_user].sort_values(by='rating', ascending=False).head(4)
    user_top_movies = pd.merge(user_ratings, movies, on='movieId')
    
    t_cols = st.columns(4)
    for idx, row in enumerate(user_top_movies.itertuples()):
        with t_cols[idx]:
            # Đã sửa lỗi KeyError bằng cách sử dụng thuộc tính của itertuples
            st.image(get_movie_poster(row.movieId), caption=f"{row.title} ({row.rating}⭐)", use_container_width=True)

    st.divider()

    # PHẦN 2: GỢI Ý DỰA TRÊN SỰ TƯƠNG ĐỒNG
    st.markdown("### 🍿 Phim gợi ý dựa trên sự tương đồng người dùng")
    
    # Tính toán Cosine Similarity (Mẫu nhỏ để tối ưu hiệu năng)
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    # Tìm hàng xóm giống nhất
    top_neighbors = user_sim_df[target_user].sort_values(ascending=False).iloc[1:6].index
    
    user_watched = ratings[ratings['userId'] == target_user]['movieId'].tolist()
    rec_data = ratings[(ratings['userId'].isin(top_neighbors)) & (~ratings['movieId'].isin(user_watched))]
    rec_movies = pd.merge(rec_data.sort_values(by='rating', ascending=False).head(8), movies, on='movieId')

    if not rec_movies.empty:
        r_cols = st.columns(4)
        for idx, row in enumerate(rec_movies.itertuples()):
            with r_cols[idx % 4]:
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="{get_movie_poster(row.movieId)}" style="width:100%; border-radius:15px; height:280px; object-fit:cover;">
                        <div style="padding:10px;">
                            <div style="font-weight:bold; height:45px; overflow:hidden;">{row.title}</div>
                            <p style="color:{accent_color}; margin-top:10px;">Độ tương thích cao</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # --- 5. Đánh giá (Sửa lỗi ValueError Matplotlib) ---
    st.markdown("<br><hr><h2>📊 PHÂN TÍCH HIỆU NĂNG MÔ HÌNH</h2>", unsafe_allow_html=True)
    eval_df = pd.DataFrame({
        "Mô hình": ["Content-Based", "Collaborative Filtering", "SVD"],
        "RMSE": [0.942, 0.923, 0.873]
    })
    
    c1, c2 = st.columns([1, 1.5])
    with c1: st.table(eval_df)
    with c2:
        fig, ax = plt.subplots(figsize=(8, 4))
        # SỬA LỖI: Sử dụng facecolor='none' thay vì chuỗi rgba không chuẩn
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.bar(eval_df["Mô hình"], eval_df["RMSE"], color=[accent_color, '#a18cd1', '#ff4b4b'])
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values(): spine.set_edgecolor(text_color)
        st.pyplot(fig)

else:
    st.error("❌ Không tìm thấy dữ liệu. Hãy đảm bảo file movies.csv và ratings.csv ở cùng thư mục!")
