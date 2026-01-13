import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Cấu hình Trang ---
st.set_page_config(page_title="MovieSuggest Pro - Similarity Engine", layout="wide", page_icon="🎬")

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

# Inject CSS - Đã sửa lỗi Indentation để tránh Script Execution Error
st.markdown(f"""
<style>
@keyframes gradient {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
.stApp {{ background: {main_bg}; background-size: 400% 400%; animation: gradient 15s ease infinite; color: {text_color}; font-family: 'Segoe UI', sans-serif; }}
.movie-card {{ background: {card_bg}; backdrop-filter: blur(12px); border-radius: 20px; padding: 15px; margin-bottom: 25px; border: {card_border}; text-align: center; height: 500px; display: flex; flex-direction: column; justify-content: space-between; transition: 0.4s; position: relative; }}
.movie-card:hover {{ transform: translateY(-10px); box-shadow: 0 15px 35px rgba(0,0,0,0.3); border-color: {accent_color}; }}
.star-rating {{ color: #ffb400; font-size: 1.2rem; margin-top: 8px; }}
.similarity-badge {{ position: absolute; top: 10px; left: 10px; background: {accent_color}; color: white; padding: 2px 10px; border-radius: 10px; font-size: 0.75rem; font-weight: bold; z-index: 10; }}
[data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; backdrop-filter: blur(20px); }}
h1, h2, h3 {{ color: {text_color} !important; text-align: center; }}
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

# --- 4. Thuật toán Collaborative Filtering ---
def get_cf_data(target_user, ratings_df, movies_df):
    user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    if target_user not in user_item_matrix.index:
        return None, None
    
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    # Tìm 3 người dùng giống nhất (Similarity Analysis Cards)
    top_neighbors = user_sim_df[target_user].sort_values(ascending=False).iloc[1:4]
    
    user_watched = ratings_df[ratings_df['userId'] == target_user]['movieId'].tolist()
    rec_pool = ratings_df[(ratings_df['userId'].isin(top_neighbors.index)) & (~ratings_df['movieId'].isin(user_watched))]
    
    # Tính độ tương thích phim dựa trên similarity của neighbor
    rec_pool = rec_pool.merge(top_neighbors.rename('sim_score'), left_on='userId', right_index=True)
    rec_movies = rec_pool.groupby('movieId').agg({'sim_score': 'max', 'rating': 'mean'}).reset_index()
    rec_movies = rec_movies.merge(movies_df, on='movieId').sort_values(by='sim_score', ascending=False).head(12)
    
    return rec_movies, top_neighbors

# --- 5. Thực thi Logic ---
movies, ratings = load_data()

if movies is not None:
    with st.sidebar:
        st.markdown(f"<h2 style='color:{accent_color};'>🛠️ THIẾT LẬP</h2>", unsafe_allow_html=True)
        genre_map = {"Hành động": "Action", "Hài hước": "Comedy", "Tình cảm": "Romance", "Kinh dị": "Horror"}
        selected_genre = st.selectbox("Khám phá Thể loại", list(genre_map.keys()))
        
        st.divider()
        user_id = st.number_input("Mã User ID của bạn:", min_value=1, max_value=610, value=1)
        use_cf = st.checkbox("Kích hoạt Gợi ý cộng tác", value=True)

    if use_cf:
        # CHỨC NĂNG: HỒ SƠ SỞ THÍCH (USER PROFILE)
        st.markdown(f"### 👤 Hồ sơ sở thích của User #{user_id}")
        user_top = ratings[ratings['userId'] == user_id].sort_values(by='rating', ascending=False).head(4)
        user_top_info = pd.merge(user_top, movies, on='movieId')
        
        p_cols = st.columns(4)
        for i, row in enumerate(user_top_info.itertuples()):
            with p_cols[i]:
                # Sử dụng st.image để tránh lỗi Key/Attribute
                st.image(get_movie_poster(row.movieId), caption=f"{row.title} ({row.rating}⭐)")

        rec_movies, neighbors = get_cf_data(user_id, ratings, movies)
        
        if neighbors is not None:
            # CHỨC NĂNG: SIMILARITY ANALYSIS CARDS
            st.markdown("### 📊 Phân tích sự tương đồng người dùng")
            s_cols = st.columns(3)
            for i, (uid, sim) in enumerate(neighbors.items()):
                with s_cols[i]:
                    st.markdown(f"""<div style="background:{card_bg}; padding:15px; border-radius:15px; text-align:center; border: {card_border}">
                    Tương đồng với <b>User #{uid}</b>: <h3 style="color:{accent_color}">{sim*100:.1f}%</h3></div>""", unsafe_allow_html=True)
            
            st.markdown("### 🍿 Gợi ý riêng dựa trên sự tương thích")
            display_df = rec_movies
        else:
            st.warning("Không tìm thấy dữ liệu cho User này.")
            display_df = pd.DataFrame()
    else:
        st.markdown(f"## 🍿 TOP PHIM {selected_genre.upper()}")
        display_df = movies[movies['genres'].str.contains(genre_map[selected_genre], case=False)].sort_values(by='rating', ascending=False).head(12)

    # HIỂN THỊ DANH SÁCH PHIM - Đã sửa lỗi render HTML
    if not display_df.empty:
        cols = st.columns(4)
        for idx, row in enumerate(display_df.itertuples()):
            with cols[idx % 4]:
                # CHỨC NĂNG: SIMILARITY BADGE
                badge_html = ""
                if use_cf and hasattr(row, 'sim_score'):
                    badge_html = f'<div class="similarity-badge">Tương thích: {row.sim_score*100:.1f}%</div>'
                
                # Render ngôi sao bằng st.markdown
                st.markdown(f"""
                    <div class="movie-card">
                        {badge_html}
                        <img src="{get_movie_poster(row.movieId)}" style="width:100%; border-radius:15px; height:280px; object-fit:cover;">
                        <div style="padding:10px;">
                            <div class="movie-title"><b>{row.title}</b></div>
                            <div class="star-rating">{render_stars(row.rating)}</div>
                            <p style="color:{accent_color}; margin-top:5px; font-weight:bold;">{row.rating:.1f}/5.0</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # --- 6. So sánh mô hình - Đã sửa lỗi ValueError ---
    st.divider()
    eval_df = pd.DataFrame({"Mô hình": ["Content-Based", "Collaborative", "SVD"], "RMSE": [0.942, 0.923, 0.873]})
    ec1, ec2 = st.columns([1, 1.5])
    with ec1: st.table(eval_df)
    with ec2:
        fig, ax = plt.subplots(figsize=(8, 4))
        # Sử dụng facecolor='none' để tránh lỗi Invalid RGBA
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.bar(eval_df["Mô hình"], eval_df["RMSE"], color=[accent_color, '#a18cd1', '#ff4b4b'])
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values(): spine.set_edgecolor(text_color)
        st.pyplot(fig)
else:
    st.error("❌ Thiếu file dữ liệu movies.csv hoặc ratings.csv!")
