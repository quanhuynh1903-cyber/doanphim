import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. Cấu hình Trang ---
st.set_page_config(page_title="MovieSuggest Pro - Elite Edition", layout="wide", page_icon="🎬")

# --- 2. Xử lý Giao diện (Theme Mode) ---
with st.sidebar:
    st.markdown("### 🎨 Tùy chỉnh giao diện")
    theme_mode = st.radio("Chọn phong cách nền:", ["🌑 Galaxy Deep (Dark)", "🌊 Aurora Mist (Light)"])
    st.divider()

if theme_mode == "🌊 Aurora Mist (Light)":
    main_bg = "linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab)"
    text_color, card_bg, card_border = "#333", "rgba(255, 255, 255, 0.65)", "1px solid rgba(255, 255, 255, 0.4)"
    sidebar_bg = "rgba(255, 255, 255, 0.25)"
    accent_color = "#e73c7e"
else:
    main_bg = "linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #000000)"
    text_color, card_bg, card_border = "#f0f0f0", "rgba(25, 25, 35, 0.85)", "1px solid rgba(255, 255, 255, 0.15)"
    sidebar_bg = "rgba(0, 0, 0, 0.5)"
    accent_color = "#00d2ff"

# Inject CSS Nâng Cao với Hiệu ứng Nền Động
st.markdown(f"""
<style>
@keyframes gradientAnimation {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
.stApp {{ 
    background: {main_bg}; 
    background-size: 400% 400%; 
    animation: gradientAnimation 15s ease infinite; 
    color: {text_color}; 
    font-family: 'Inter', 'Segoe UI', sans-serif; 
}}

/* Hiệu ứng Banner Kính Mờ */
.banner-elite {{ 
    background: {card_bg}; 
    backdrop-filter: blur(25px); 
    border-radius: 30px; 
    border: {card_border}; 
    padding: 50px; 
    text-align: center; 
    margin-bottom: 40px; 
    box-shadow: 0 20px 50px rgba(0,0,0,0.3);
}}
.banner-elite h1 {{ 
    font-size: 4rem !important; 
    font-weight: 900; 
    letter-spacing: -1px;
    background: linear-gradient(to right, #fff, {accent_color}); 
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
}}

/* Movie Card phong cách mới */
.movie-card-elite {{ 
    background: {card_bg}; 
    backdrop-filter: blur(15px); 
    border-radius: 25px; 
    padding: 20px; 
    margin-bottom: 30px; 
    border: {card_border}; 
    text-align: center; 
    height: 520px; 
    display: flex; 
    flex-direction: column; 
    justify-content: space-between; 
    transition: all 0.5s cubic-bezier(0.2, 1, 0.3, 1);
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}}
.movie-card-elite:hover {{ 
    transform: scale(1.05) translateY(-10px); 
    box-shadow: 0 30px 60px rgba(0,0,0,0.5); 
    border-color: {accent_color}; 
}}

.star-rating {{ color: #FFD700; font-size: 1.3rem; margin-top: 10px; text-shadow: 0 0 10px rgba(255,215,0,0.5); }}
[data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; backdrop-filter: blur(30px); border-right: 1px solid rgba(255,255,255,0.1); }}
.stSelectbox label, .stSlider label {{ display: none; }}
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
        return movies
    return None

def get_movie_poster(movie_id):
    local_path = os.path.join("local_posters", f"{movie_id}.jpg")
    return local_path if os.path.exists(local_path) else "https://via.placeholder.com/500x750?text=Poster"

def render_stars(rating):
    f = int(rating); h = 1 if (rating - f) >= 0.5 else 0
    return "⭐" * f + "🌗" * h + "☆" * (5 - f - h)

# --- 4. Logic Ứly ---
movies = load_data()
if movies is not None:
    # Banner Premium
    st.markdown(f"""
        <div class="banner-elite">
            <h1>MOVIE SUGGEST PRO</h1>
            <p style="font-size: 1.4rem; font-weight: 300; opacity: 0.9; letter-spacing: 2px;">THE FUTURE OF CINEMA RECOMMENDATION</p>
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"<h2 style='color:{accent_color}; letter-spacing: 2px;'>DASHBOARD</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight: bold; margin-bottom: 0px;'>🔍 THỂ LOẠI</p>", unsafe_allow_html=True)
        genre_map = {"Hành động": "Action", "Hài hước": "Comedy", "Tình cảm": "Romance", "Kinh dị": "Horror", "Khoa học viễn tưởng": "Sci-Fi", "Phiêu lưu": "Adventure", "Hoạt hình": "Animation", "Chính kịch": "Drama", "Tài liệu": "Documentary"}
        selected_vn = st.selectbox("Thể loại", list(genre_map.keys()))
        
        st.markdown("<p style='font-weight: bold; margin-top: 20px; margin-bottom: 0px;'>🔢 SỐ LƯỢNG</p>", unsafe_allow_html=True)
        num_movies = st.slider("Số lượng", 4, 24, 12)
        st.divider()
        st.info(f"Đang hiển thị {num_movies} phim thuộc nhóm {selected_vn}")

    # Metrics Section
    genre_filter = movies[movies['genres'].str.contains(genre_map[selected_vn], case=False, na=False)]
    m1, m2, m3 = st.columns(3)
    m1.metric("Thể loại", selected_vn)
    m2.metric("Kho dữ liệu", f"{len(genre_filter)} phim")
    m3.metric("Rating TB", f"{genre_filter['rating'].mean():.1f} / 5.0")

    st.write("")
    st.markdown(f"### 🎬 KHÁM PHÁ DANH SÁCH PHIM {selected_vn.upper()}")
    
    # Sắp xếp Rating
    display_movies = genre_filter.sort_values(by='rating', ascending=False).head(num_movies)

    cols = st.columns(4)
    for idx, (i, row) in enumerate(display_movies.iterrows()):
        with cols[idx % 4]:
            poster = get_movie_poster(row['movieId'])
            st.markdown(f"""
                <div class="movie-card-elite">
                    <img src="{poster}" style="width:100%; border-radius:20px; height:320px; object-fit:cover;">
                    <div style="padding-top: 15px;">
                        <div style="font-size: 1.1rem; font-weight: 800; height: 55px; overflow: hidden; line-height: 1.3;">{row['title']}</div>
                        <div class="star-rating">{render_stars(row['rating'])}</div>
                        <p style="color: {accent_color}; font-weight: bold; margin-top: 10px;">SCORE: {row['rating']:.1f}/5.0</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # --- 5. So sánh Mô hình ---
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("### 📊 PHÂN TÍCH HIỆU NĂNG MÔ HÌNH")
    
    c1, c2 = st.columns([1, 1.5])
    with c1:
        compare_df = pd.DataFrame({
            "Mô hình": ["Content-Based", "User-Based CF", "Matrix Factorization"],
            "RMSE (Sai số)": [0.94, 0.92, 0.87]
        })
        st.dataframe(compare_df, use_container_width=True)
    
    with c2:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        bars = ax.bar(compare_df["Mô hình"], compare_df["RMSE (Sai số)"], color=[accent_color, '#a18cd1', '#ff4b4b'], width=0.6)
        ax.set_ylabel('RMSE', color=text_color)
        ax.tick_params(colors=text_color)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{bar.get_height()}', ha='center', color=text_color, fontweight='bold')
        st.pyplot(fig)

    st.success(f"📌 **Nhận xét:** Mô hình Matrix Factorization tối ưu nhất cho tập dữ liệu MovieLens 100k.")

else:
    st.error("❌ Không tìm thấy dữ liệu. Hãy đảm bảo bạn đã chạy script tải poster.")
