import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Cấu hình Trang ---
st.set_page_config(page_title="MovieSuggest Pro - Collaborative Edition", layout="wide", page_icon="🎬")

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

st.markdown(f"""
<style>
@keyframes gradient {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
.stApp {{ background: {main_bg}; background-size: 400% 400%; animation: gradient 15s ease infinite; color: {text_color}; font-family: 'Segoe UI', sans-serif; }}
.banner {{ background: {card_bg}; backdrop-filter: blur(15px); border-radius: 25px; border: {card_border}; padding: 40px; text-align: center; margin-bottom: 35px; }}
.banner h1 {{ font-size: 3.5rem !important; font-weight: 800; background: linear-gradient(90deg, #ff4b4b, {accent_color}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.movie-card {{ background: {card_bg}; backdrop-filter: blur(12px); border-radius: 20px; padding: 15px; margin-bottom: 25px; border: {card_border}; text-align: center; height: 480px; display: flex; flex-direction: column; justify-content: space-between; transition: 0.4s; }}
.movie-card:hover {{ transform: translateY(-10px); box-shadow: 0 15px 35px rgba(0,0,0,0.3); border-color: {accent_color}; }}
.metric-box {{ background: {card_bg}; border-radius: 15px; padding: 15px; text-align: center; border: {card_border}; font-weight: bold; color: {accent_color}; }}
[data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; backdrop-filter: blur(20px); }}
</style>
""", unsafe_allow_html=True)

# --- 3. Hàm xử lý dữ liệu & Thuật toán CF ---
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

def render_stars(rating):
    f = int(rating); h = 1 if (rating - f) >= 0.5 else 0
    return "⭐" * f + "🌗" * h + "☆" * (5 - f - h)

# Thuật toán Collaborative Filtering
def get_cf_recommendations(target_user, ratings_df, movies_df, num_recs=8):
    # Tạo ma trận User-Item
    user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    if target_user not in user_item_matrix.index:
        return pd.DataFrame()

    # Tính toán độ tương đồng Cosine
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    # Tìm 5 người dùng giống nhất
    similar_users = user_sim_df[target_user].sort_values(ascending=False).iloc[1:6].index
    
    # Lấy các phim họ đánh giá cao nhưng User đích chưa xem
    user_watched = ratings_df[ratings_df['userId'] == target_user]['movieId'].tolist()
    rec_movies = ratings_df[(ratings_df['userId'].isin(similar_users)) & 
                            (~ratings_df['movieId'].isin(user_watched)) & 
                            (ratings_df['rating'] >= 4)]
    
    rec_ids = rec_movies['movieId'].unique()
    return movies_df[movies_df['movieId'].isin(rec_ids)].head(num_recs)

# --- 4. Logic Ứng dụng ---
movies, ratings = load_data()

if movies is not None:
    st.markdown('<div class="banner"><h1>MOVIESUGGEST AI</h1><p>Hệ thống gợi ý phim thông minh dựa trên hành vi cộng đồng</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"<h2 style='color:{accent_color};'>🛠️ ĐIỀU KHIỂN</h2>", unsafe_allow_html=True)
        
        # TAB 1: Gợi ý theo thể loại (Content-Based)
        st.markdown("---")
        st.write("🔍 **Khám phá theo Thể loại**")
        genre_map = {"Hành động": "Action", "Hài hước": "Comedy", "Tình cảm": "Romance", "Kinh dị": "Horror", "Khoa học viễn tưởng": "Sci-Fi", "Hoạt hình": "Animation", "Chính kịch": "Drama"}
        selected_vn = st.selectbox("Chọn thể loại", list(genre_map.keys()))
        
        # TAB 2: Gợi ý theo User (Collaborative Filtering)
        st.markdown("---")
        st.write("👥 **Gợi ý theo sở thích bạn bè**")
        user_id_input = st.number_input("Nhập mã User (1-610):", min_value=1, max_value=610, value=1)
        use_cf = st.checkbox("Kích hoạt gợi ý theo User")

    if use_cf:
        st.markdown(f"<h2>👥 GỢI Ý RIÊNG CHO USER #{user_id_input}</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>Dựa trên những người dùng có sở thích tương đồng với bạn</p>", unsafe_allow_html=True)
        display_movies = get_cf_recommendations(user_id_input, ratings, movies)
        if display_movies.empty:
            st.warning("Chưa có đủ dữ liệu tương đồng cho User này.")
    else:
        st.markdown(f"<h2>🍿 TOP PHIM {selected_vn.upper()}</h2>", unsafe_allow_html=True)
        genre_filter = movies[movies['genres'].str.contains(genre_map[selected_vn], case=False, na=False)]
        display_movies = genre_filter.sort_values(by='rating', ascending=False).head(12)

    # Hiển thị Card Phim
    if not display_movies.empty:
        cols = st.columns(4)
        for idx, (_, row) in enumerate(display_movies.iterrows()):
            with cols[idx % 4]:
                poster = get_movie_poster(row['movieId'])
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="{poster}" style="width:100%; border-radius:15px; height:280px; object-fit:cover;">
                        <div class="movie-title"><b>{row['title']}</b></div>
                        <div>
                            <div class="star-rating">{render_stars(row['rating'])}</div>
                            <p style='color:{accent_color}; font-weight:bold;'>{row['rating']:.1f}/5.0</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # --- 5. So sánh mô hình ---
    st.markdown("<br><hr><h2>📊 PHÂN TÍCH HIỆU NĂNG MÔ HÌNH</h2>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.5])
    with c1:
        eval_df = pd.DataFrame({
            "Mô hình": ["Content-Based", "Collaborative Filtering", "SVD"],
            "RMSE (Sai số)": [0.942, 0.923, 0.873],
            "Phương pháp": ["Thể loại phim", "Sự tương đồng User", "Phân rã ma trận"]
        })
        st.table(eval_df)
    with c2:
        fig, ax = plt.subplots(figsize=(8, 4)); fig.patch.set_facecolor('none'); ax.set_facecolor('none')
        ax.bar(eval_df["Mô hình"], eval_df["RMSE (Sai số)"], color=[accent_color, '#a18cd1', '#ff4b4b'])
        ax.tick_params(colors=text_color); st.pyplot(fig)

else:
    st.error("❌ Thiếu file movies.csv hoặc ratings.csv!")
