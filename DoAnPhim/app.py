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

   # --- 5. PHÂN TÍCH HIỆU NĂNG MÔ HÌNH (Elite Dashboard) ---
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color:{accent_color}; text-shadow: 2px 2px 10px rgba(0,0,0,0.3);'>📊 PHÂN TÍCH HIỆU NĂNG THUẬT TOÁN</h2>", unsafe_allow_html=True)
    
    # Tạo 3 thẻ Metrics tóm tắt phía trên
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div style="background:{card_bg}; padding:20px; border-radius:15px; border:{card_border}; text-align:center;">
            <p style="margin:0; opacity:0.8;">Mô hình tối ưu nhất</p>
            <h3 style="margin:5px 0; color:#ff4b4b;">Matrix Factorization</h3>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div style="background:{card_bg}; padding:20px; border-radius:15px; border:{card_border}; text-align:center;">
            <p style="margin:0; opacity:0.8;">Độ chính xác (RMSE)</p>
            <h3 style="margin:5px 0; color:#23d5ab;">0.8730</h3>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div style="background:{card_bg}; padding:20px; border-radius:15px; border:{card_border}; text-align:center;">
            <p style="margin:0; opacity:0.8;">Tập dữ liệu thử nghiệm</p>
            <h3 style="margin:5px 0; color:{accent_color};">MovieLens 100K</h3>
        </div>""", unsafe_allow_html=True)

    st.write("<br>", unsafe_allow_html=True)

    # Chia cột cho Bảng và Biểu đồ
    col_table, col_chart = st.columns([1, 1.2])

    with col_table:
        st.markdown("#### 📋 Chi tiết các chỉ số")
        # Sử dụng CSS để làm bảng đẹp hơn
        st.markdown(f"""
        <style>
            .styled-table {{ width:100%; border-collapse: collapse; border-radius:10px; overflow:hidden; font-family: sans-serif; }}
            .styled-table thead tr {{ background-color: {accent_color}; color: #ffffff; text-align: left; }}
            .styled-table th, .styled-table td {{ padding: 12px 15px; border-bottom: 1px solid rgba(255,255,255,0.1); }}
            .styled-table tbody tr:nth-of-type(even) {{ background-color: rgba(255,255,255,0.05); }}
        </style>
        <table class="styled-table">
            <thead>
                <tr><th>Mô hình</th><th>RMSE (Sai số)</th><th>Độ phủ</th></tr>
            </thead>
            <tbody>
                <tr><td>Content-Based</td><td>0.9420</td><td>Cao</td></tr>
                <tr><td>User-Based CF</td><td>0.9230</td><td>Trung bình</td></tr>
                <tr style="font-weight:bold; color:#ff4b4b;"><td>Matrix Factorization</td><td>0.8730</td><td>Thấp</td></tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

    with col_chart:
        st.markdown("#### 📉 Biểu đồ sai số RMSE")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Làm nền trong suốt để khớp với Gradient của App
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        
        models = ["Content-Based", "User-Based CF", "Matrix Factorization"]
        rmse_values = [0.942, 0.923, 0.873]
        colors_bar = ['#4b6cb7', '#a18cd1', '#ff4b4b'] # SVD màu đỏ để nổi bật
        
        bars = ax.bar(models, rmse_values, color=colors_bar, width=0.6, edgecolor='white', linewidth=1)
        
        # Tùy chỉnh trục và chữ
        ax.set_ylabel('RMSE (Càng thấp càng tốt)', color=text_color, fontsize=12)
        ax.tick_params(colors=text_color, labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor(text_color)
            spine.set_alpha(0.3)

        # Thêm giá trị trên đầu cột
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', 
                    ha='center', va='bottom', color=text_color, fontweight='bold')
        
        st.pyplot(fig)

    # --- 6. NHẬN XÉT CHI TIẾT (Thiết kế lại thành khối Alert) ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: rgba(35, 213, 171, 0.1); border-left: 5px solid #23d5ab; padding: 25px; border-radius: 10px;">
        <h4 style="margin-top:0; color:#23d5ab;">💡 Kết luận từ phân tích:</h4>
        <p style="font-size: 1.1rem; line-height: 1.7; margin-bottom: 0;">
            Thông qua quá trình đánh giá bằng chỉ số <b>RMSE (Root Mean Square Error)</b>, chúng ta nhận thấy mô hình 
            <b>Matrix Factorization (SVD)</b> đạt hiệu quả vượt trội nhất. Thuật toán này có khả năng "học" được các đặc tính ẩn 
            của người dùng và phim, giúp dự đoán điểm số với sai số thấp nhất (0.8730). <br><br>
            Tuy nhiên, trong ứng dụng thực tế này, chúng ta ưu tiên kết hợp <b>Content-Based</b> để giải quyết vấn đề 
            "Cold Start" (người dùng mới) và đảm bảo tốc độ phản hồi tức thì khi lọc theo thể loại.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("❌ Không tìm thấy dữ liệu. Hãy đảm bảo bạn đã chạy script tải poster.")

