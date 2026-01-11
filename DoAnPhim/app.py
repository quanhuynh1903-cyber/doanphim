import streamlit as st
import pandas as pd
import os

# --- Cấu hình Trang ---
st.set_page_config(page_title="MovieSuggest - Khám Phá Điện Ảnh", layout="wide", page_icon="🎬")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .movie-card {
        background-color: #1c2128;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #30363d;
        text-align: center;
        height: 350px;
        transition: 0.3s;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .movie-card:hover {
        border-color: #58a6ff;
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }
    .movie-title {
        color: #adbac7;
        font-size: 1rem;
        font-weight: bold;
        margin-top: 10px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .genre-tag {
        background-color: #2d333b;
        color: #58a6ff;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 10px;
        margin-top: 5px;
        display: inline-block;
    }
    h1, h2, h3 { color: #58a6ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Tải dữ liệu ---
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    movies_path = os.path.join(base_path, 'movies.csv')
    ratings_path = os.path.join(base_path, 'ratings.csv')
    if not os.path.exists(movies_path):
        return None
    
    movies = pd.read_csv(movies_path)
    # Tính điểm trung bình cho mỗi phim nếu có file ratings
    if os.path.exists(ratings_path):
        ratings = pd.read_csv(ratings_path)
        avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
        movies = pd.merge(movies, avg_ratings, on='movieId', how='left')
    else:
        movies['rating'] = 0
    return movies

movies = load_data()

if movies is not None:
    # --- SIDEBAR: Chọn sở thích ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2503/2503508.png", width=80)
        st.title("Sở Thích")
        st.write("Chọn thể loại bạn muốn xem hôm nay:")
        
        # Danh sách thể loại theo yêu cầu của bạn
        genre_options = [
            "Action", "Comedy", "Romance", "Horror", "Sci-Fi", 
            "Adventure", "Animation", "Drama", "Documentary", "Fantasy"
        ]
        
        # Ánh xạ tên tiếng Việt sang tiếng Anh (vì dữ liệu MovieLens dùng tiếng Anh)
        genre_map = {
            "Hành động": "Action",
            "Hài hước": "Comedy",
            "Tình cảm": "Romance",
            "Kinh dị": "Horror",
            "Khoa học viễn tưởng": "Sci-Fi",
            "Phiêu lưu": "Adventure",
            "Hoạt hình": "Animation",
            "Chính kịch": "Drama",
            "Tài liệu": "Documentary"
        }
        
        selected_vn = st.selectbox("Chọn thể loại:", list(genre_map.keys()))
        selected_genre = genre_map[selected_vn]
        
        num_movies = st.slider("Số lượng phim đề xuất:", 4, 20, 12)
        
        st.divider()
        st.caption("Dữ liệu dựa trên cộng đồng MovieLens")

    # --- NỘI DUNG CHÍNH ---
    st.markdown(f"<h1>🎬 Đề Xuất Phim {selected_vn}</h1>", unsafe_allow_html=True)
    st.write(f"Tìm thấy những bộ phim phù hợp nhất với phong cách **{selected_vn}** của bạn.")

    # Lọc phim theo thể loại và sắp xếp theo rating (nếu có)
    filtered_movies = movies[movies['genres'].str.contains(selected_genre, case=False, na=False)]
    filtered_movies = filtered_movies.sort_values(by='rating', ascending=False).head(num_movies)

    if not filtered_movies.empty:
        # Hiển thị Grid phim
        cols = st.columns(4)
        for idx, (_, row) in enumerate(filtered_movies.iterrows()):
            with cols[idx % 4]:
                # Hiển thị card phim
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="https://via.placeholder.com/200x250/2d333b/ffffff?text={selected_vn}" style="width:100%; border-radius:8px;">
                        <div>
                            <div class="movie-title">{row['title']}</div>
                            <div class="genre-tag">{selected_vn}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                # Nút "Chi tiết" thay cho "Xem ngay"
                if st.button("📄 Thông tin", key=f"info_{row['movieId']}"):
                    st.info(f"**{row['title']}**\n\nThể loại đầy đủ: {row['genres']}\n\nĐánh giá trung bình: {row['rating']:.1f}/5⭐")
    else:
        st.warning("Rất tiếc, không tìm thấy phim thuộc thể loại này trong kho dữ liệu.")

    # --- KHÁM PHÁ THÊM ---
    st.divider()
    st.subheader("✨ Có thể bạn cũng thích")
    random_suggest = movies.sample(4)
    r_cols = st.columns(4)
    for idx, (_, row) in enumerate(random_suggest.iterrows()):
        with r_cols[idx]:
            st.image("https://via.placeholder.com/150x200/161b22/58a6ff?text=SURPRISE", caption=row['title'][:20])
else:
    st.error("Không tìm thấy file movies.csv. Vui lòng kiểm tra thư mục dự án.")
