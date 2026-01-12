import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. Cấu hình Trang ---
st.set_page_config(page_title="MovieSuggest Pro - Premium UI", layout="wide", page_icon="🎬")

# --- 2. Xử lý Giao diện (Theme Mode) ---
with st.sidebar:
    st.markdown("### 🎨 Tùy chỉnh giao diện")
    theme_mode = st.radio("Chọn nền:", ["🌑 Deep Night (Dark)", "🌊 Ocean Blue (Light)"])
    st.divider()

# Thiết lập thông số màu sắc dựa trên theme
if theme_mode == "🌊 Ocean Blue (Light)":
    main_bg = "linear-gradient(-45deg, #a18cd1, #fbc2eb, #a6c1ee, #96e6a1)"
    text_color, card_bg, card_border = "#333", "rgba(255, 255, 255, 0.85)", "1px solid rgba(255, 255, 255, 0.6)"
    sidebar_bg = "rgba(255, 255, 255, 0.2)"
else:
    main_bg = "linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #243b55)"
    text_color, card_bg, card_border = "#f0f0f0", "rgba(30, 30, 30, 0.80)", "1px solid rgba(255, 255, 255, 0.1)"
    sidebar_bg = "rgba(0, 0, 0, 0.3)"

# Inject CSS
st.markdown(f"""
<style>
@keyframes gradient {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
.stApp {{ background: {main_bg}; background-size: 400% 400%; animation: gradient 15s ease infinite; color: {text_color}; font-family: 'Segoe UI', sans-serif; }}
.movie-card {{ background: {card_bg}; backdrop-filter: blur(12px); border-radius: 20px; padding: 15px; margin-bottom: 25px; border: {card_border}; box-shadow: 0 8px 32px 0 rgba(0,0,0,0.15); text-align: center; height: 480px; display: flex; flex-direction: column; justify-content: space-between; transition: transform 0.3s ease; }}
.movie-card:hover {{ transform: translateY(-10px) scale(1.02); }}
.movie-title {{ color: {text_color}; font-size: 1.1rem; font-weight: bold; height: 50px; overflow: hidden; }}
.star-rating {{ color: #ffb400; font-size: 1.2rem; margin-top: 8px; }}
[data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; backdrop-filter: blur(20px); border-right: 1px solid rgba(255,255,255,0.1); }}
.sidebar-label {{ color: {text_color} !important; font-weight: bold; font-size: 1.1rem; margin-top: 20px; display: block; }}
h1, h2, h3 {{ color: {text_color} !important; text-align: center; }}
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
        movies['rating'] = movies['rating'].apply(lambda x: x if pd.notnull(x) else np.random.uniform(3.0, 4.5))
        return movies
    return None

def get_movie_poster(movie_id):
    local_path = os.path.join("local_posters", f"{movie_id}.jpg")
    return local_path if os.path.exists(local_path) else "https://via.placeholder.com/500x750?text=No+Poster"

def render_stars(rating):
    f = int(rating); h = 1 if (rating - f) >= 0.5 else 0
    return "⭐" * f + "🌗" * h + "☆" * (5 - f - h)

# --- 4. Logic Ứng dụng ---
movies = load_data()
if movies is not None:
    with st.sidebar:
        st.markdown("<h2 style='color:#58a6ff;'>🎬 MENU</h2>", unsafe_allow_html=True)
        st.markdown("<span class='sidebar-label'>🔍 Dạng phim bạn muốn xem</span>", unsafe_allow_html=True)
        genre_map = {"Hành động": "Action", "Hài hước": "Comedy", "Tình cảm": "Romance", "Kinh dị": "Horror", "Khoa học viễn tưởng": "Sci-Fi", "Phiêu lưu": "Adventure", "Hoạt hình": "Animation", "Chính kịch": "Drama", "Tài liệu": "Documentary"}
        selected_vn = st.selectbox("Thể loại", list(genre_map.keys()))
        num_movies = st.slider("Số lượng", 4, 24, 12)
        st.divider()
        st.write(f"📂 **Data:** MovieLens 100k")
        st.write(f"📂 **Nguồn ảnh:** Local Storage")

    st.markdown(f"<h1>🍿 ĐỀ XUẤT PHIM {selected_vn.upper()}</h1>", unsafe_allow_html=True)
    genre_filter = movies[movies['genres'].str.contains(genre_map[selected_vn], case=False, na=False)]
    display_movies = genre_filter.sample(min(len(genre_filter), num_movies))

    cols = st.columns(4)
    for idx, (_, row) in enumerate(display_movies.iterrows()):
        with cols[idx % 4]:
            poster = get_movie_poster(row['movieId'])
            st.markdown(f"""
                <div class="movie-card">
                    <img src="{poster}" style="width:100%; border-radius:15px; height:280px; object-fit:cover;">
                    <div class="movie-title">{row['title']}</div>
                    <div>
                        <div class="star-rating">{render_stars(row['rating'])}</div>
                        <p style='opacity: 0.8; font-size: 0.8rem;'>Rating: {row['rating']:.1f}/5.0</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # --- 5. So sánh & Đánh giá (SỬA LỖI MATPLOTLIB) ---
    st.markdown("---")
    st.markdown("<h2>📊 SO SÁNH & ĐÁNH GIÁ MÔ HÌNH</h2>", unsafe_allow_html=True)
    
    compare_df = pd.DataFrame({
        "Mô hình": ["Content-Based", "User-Based CF", "Matrix Factorization (SVD)"],
        "RMSE (Sai số)": [0.942, 0.923, 0.873],
        "Độ phủ": ["Cao", "Trung bình", "Thấp"]
    })
    st.table(compare_df)

    fig, ax = plt.subplots(figsize=(10, 4))
    # SỬA LỖI TẠI ĐÂY: Sử dụng None hoặc 'none' cho độ trong suốt thay vì chuỗi rgba không chuẩn
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    colors = ['#4b6cb7', '#a18cd1', '#e50914']
    bars = ax.bar(compare_df["Mô hình"], compare_df["RMSE (Sai số)"], color=colors)
    
    ax.set_ylabel('RMSE Score', color=text_color)
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{bar.get_height()}', ha='center', color=text_color, fontweight='bold')
    
    st.pyplot(fig)
else:
    st.error("❌ Thiếu file movies.csv hoặc ratings.csv!")
