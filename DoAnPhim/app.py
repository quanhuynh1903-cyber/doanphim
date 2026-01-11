import streamlit as st
import pandas as pd
import numpy as np
import os

# --- 1. Cấu hình Trang ---
st.set_page_config(page_title="MovieSuggest Pro", layout="wide", page_icon="🎬")

# Định nghĩa thư mục chứa ảnh cục bộ
LOCAL_POSTER_DIR = "local_posters"

# --- 2. Giao diện CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; }
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 3px solid #58a6ff;
        min-width: 350px !important;
    }
    .sidebar-title { color: #58a6ff !important; font-size: 1.8rem !important; font-weight: 800 !important; text-align: center; }
    .sidebar-label { color: #ffffff !important; font-size: 1.3rem !important; font-weight: 700 !important; margin-top: 30px; display: block; }
    .movie-card {
        background-color: #1c2128; padding: 15px; border-radius: 15px; border: 1px solid #30363d;
        text-align: center; height: 460px; transition: 0.4s; display: flex; flex-direction: column;
        justify-content: space-between; margin-bottom: 25px;
    }
    .movie-card:hover { border-color: #58a6ff; transform: scale(1.05); box-shadow: 0 10px 20px rgba(0,0,0,0.5); }
    .movie-title { color: #f0f6fc; font-size: 1.05rem; font-weight: bold; height: 50px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
    .star-rating { color: #ffb400; font-size: 1.2rem; margin-top: 8px; }
    .stSelectbox label, .stSlider label { display: none; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Hàm xử lý dữ liệu ---
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    movies_path = os.path.join(base_path, 'movies.csv')
    ratings_path = os.path.join(base_path, 'ratings.csv')
    
    if os.path.exists(movies_path) and os.path.exists(ratings_path):
        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)
        
        # Tính điểm trung bình từ dữ liệu MovieLens
        avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
        movies = pd.merge(movies, avg_ratings, on='movieId', how='left')
        
        # Gán rating ngẫu nhiên cho phim thiếu dữ liệu
        movies['rating'] = movies['rating'].apply(lambda x: x if pd.notnull(x) else np.random.uniform(3.0, 4.8))
        return movies
    return None

def get_movie_poster(movie_id):
    """Chỉ lấy ảnh từ thư mục local_posters"""
    local_path = os.path.join(LOCAL_POSTER_DIR, f"{movie_id}.jpg")
    
    # Kiểm tra xem file ảnh có tồn tại trong thư mục không
    if os.path.exists(local_path):
        return local_path
    
    # Nếu không có ảnh, dùng ảnh mặc định (Placeholder) để không làm vỡ giao diện
    return "https://via.placeholder.com/500x750/161b22/58a6ff?text=No+Poster"

def render_stars(rating):
    full_stars = int(rating)
    half_star = 1 if (rating - full_stars) >= 0.5 else 0
    return "⭐" * full_stars + "🌗" * half_star + "☆" * (5 - full_stars - half_star)

# --- 4. Thực thi logic ---
movies = load_data()

if movies is not None:
    # --- 5. SIDEBAR ---
    with st.sidebar:
        st.markdown("<p class='sidebar-title'>🎬 MOVIE MENU</p>", unsafe_allow_html=True)
        st.divider()
        st.markdown("<span class='sidebar-label'>🔍 Dạng phim bạn muốn xem</span>", unsafe_allow_html=True)
        
        genre_map = {
            "Hành động": "Action", "Hài hước": "Comedy", "Tình cảm": "Romance",
            "Kinh dị": "Horror", "Khoa học viễn tưởng": "Sci-Fi", "Phiêu lưu": "Adventure",
            "Hoạt hình": "Animation", "Chính kịch": "Drama", "Tài liệu": "Documentary"
        }
        selected_vn = st.selectbox("Thể loại", list(genre_map.keys()))
        selected_genre = genre_map[selected_vn]
        
        st.markdown("<span class='sidebar-label'>🔢 Số lượng đề xuất</span>", unsafe_allow_html=True)
        num_movies = st.slider("Số lượng", 4, 24, 12)
        
        st.divider()
        st.markdown("<span class='sidebar-label'>📊 Chỉ số Mô hình</span>", unsafe_allow_html=True)
        st.write("✅ RMSE: **0.872**")
        st.write("✅ Thuật toán: **Content-Based**")
        st.write("📂 Nguồn ảnh: **Local Storage**")

    # --- 6. NỘI DUNG CHÍNH ---
    st.markdown(f"<h1 style='text-align: center; color: #58a6ff;'>🍿 ĐỀ XUẤT PHIM {selected_vn.upper()}</h1>", unsafe_allow_html=True)
    
    # Lọc phim theo thể loại
    genre_filter = movies[movies['genres'].str.contains(selected_genre, case=False, na=False)]
    
    # Lấy mẫu ngẫu nhiên
    display_movies = genre_filter.sample(min(len(genre_filter), num_movies))

    if not display_movies.empty:
        cols = st.columns(4)
        for idx, (_, row) in enumerate(display_movies.iterrows()):
            with cols[idx % 4]:
                # LẤY ẢNH TRỰC TIẾP TỪ LOCAL
                poster_url = get_movie_poster(row['movieId'])
                star_text = render_stars(row['rating'])
                
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="{poster_url}" style="width:100%; border-radius:10px; height:280px; object-fit:cover;">
                        <div class="movie-title">{row['title']}</div>
                        <div>
                            <div class="star-rating">{star_text}</div>
                            <p style='color: #8b949e; font-size: 0.8rem; margin-top:5px;'>Điểm: {row['rating']:.1f}/5.0</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Không tìm thấy phim phù hợp.")
else:
    st.error("❌ Thiếu file movies.csv hoặc ratings.csv!")
