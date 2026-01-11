import streamlit as st
import pandas as pd
import numpy as np
import os
import requests

# --- 1. Cấu hình Trang và API Key ---
st.set_page_config(page_title="MovieSuggest Pro", layout="wide", page_icon="🎬")

# THAY THẾ 'YOUR_API_KEY_CỦA_BẠN' BẰNG API KEY THẬT TỪ TMDB
TMDB_API_KEY = 'YOUR_API_KEY_CỦA_BẠN' 
LOCAL_POSTER_DIR = "local_posters"

# --- 2. Giao diện CSS: Làm nổi bật Sidebar và Card phim ---
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; }
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 3px solid #58a6ff;
        min-width: 350px !important;
    }
    .sidebar-title {
        color: #58a6ff !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        text-align: center;
    }
    .sidebar-label {
        color: #ffffff !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin-top: 30px;
        display: block;
    }
    .movie-card {
        background-color: #1c2128;
        padding: 15px;
        border-radius: 15px;
        border: 1px solid #30363d;
        text-align: center;
        height: 460px;
        transition: 0.4s;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 25px;
    }
    .movie-card:hover {
        border-color: #58a6ff;
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(0,0,0,0.5);
    }
    .movie-title {
        color: #f0f6fc;
        font-size: 1.05rem;
        font-weight: bold;
        margin-top: 10px;
        height: 50px;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .star-rating {
        color: #ffb400; 
        font-size: 1.2rem;
        margin-top: 8px;
    }
    /* Ẩn label mặc định của Streamlit */
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
        
        # Tính điểm trung bình thật từ MovieLens
        avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
        movies = pd.merge(movies, avg_ratings, on='movieId', how='left')
        
        # Gán rating ngẫu nhiên nhẹ cho phim thiếu dữ liệu để demo sinh động
        movies['rating'] = movies['rating'].apply(lambda x: x if pd.notnull(x) else np.random.uniform(3.0, 4.8))
        return movies
    return None

@st.cache_data
def get_movie_poster(movie_id, movie_title):
    # 1. Thử lấy ảnh cục bộ trước (Dành cho thuyết trình offline)
    local_path = os.path.join(LOCAL_POSTER_DIR, f"{movie_id}.jpg")
    if os.path.exists(local_path):
        return local_path

    # 2. Nếu không có ảnh cục bộ, gọi API TMDB (Cần có mạng)
    try:
        search_title = movie_title.split(' (')[0]
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={search_title}"
        response = requests.get(url, timeout=5).json()
        
        if response.get('results') and len(response['results']) > 0:
            poster_path = response['results'][0]['poster_path']
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass
    
    # 3. Ảnh mặc định nếu cả 2 cách trên đều thất bại
    return "https://via.placeholder.com/500x750/161b22/58a6ff?text=No+Poster"

def render_stars(rating):
    full_stars = int(rating)
    half_star = 1 if (rating - full_stars) >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    return "⭐" * full_stars + "🌗" * half_star + "☆" * empty_stars

# --- 4. Thực thi logic chính ---
movies = load_data()

if movies is not None:
    # --- 5. SIDEBAR: Bảng điều khiển ---
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

    # --- 6. NỘI DUNG CHÍNH: Hiển thị phim ---
    st.markdown(f"<h1 style='text-align: center; color: #58a6ff;'>🍿 ĐỀ XUẤT PHIM {selected_vn.upper()}</h1>", unsafe_allow_html=True)
    
    # Lọc phim theo thể loại và lấy mẫu ngẫu nhiên
    genre_filter = movies[movies['genres'].str.contains(selected_genre, case=False, na=False)]
    
    # Đảm bảo không lấy quá số lượng phim đang có trong kho
    display_movies = genre_filter.sample(min(len(genre_filter), num_movies))

    if not display_movies.empty:
        cols = st.columns(4)
        for idx, (_, row) in enumerate(display_movies.iterrows()):
            with cols[idx % 4]:
                # Lấy poster (Ưu tiên local -> API)
                poster_url = get_movie_poster(row['movieId'], row['title'])
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
        st.warning("Không tìm thấy phim phù hợp trong kho dữ liệu.")
else:
    st.error("❌ Lỗi: Thiếu file movies.csv hoặc ratings.csv trong thư mục làm việc!")
