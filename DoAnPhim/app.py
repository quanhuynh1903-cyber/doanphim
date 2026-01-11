import streamlit as st
import pandas as pd
import os

# --- 1. Cấu hình Trang ---
st.set_page_config(page_title="MovieSuggest Pro", layout="wide", page_icon="🎬")

# --- 2. CUSTOM CSS: Làm nổi bật Sidebar và chữ ---
st.markdown("""
    <style>
    /* Nền tối cho trang */
    .stApp { background-color: #0d1117; }
    
    /* Sidebar nổi bật hoàn toàn */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 3px solid #58a6ff; /* Viền xanh nổi bật */
        min-width: 350px !important;
    }

    /* Làm chữ trong Sidebar cực kỳ rõ ràng */
    .sidebar-title {
        color: #58a6ff !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        margin-bottom: 20px;
        text-transform: uppercase;
        text-align: center;
    }
    
    .sidebar-label {
        color: #ffffff !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin-top: 25px;
        display: block;
    }

    /* Thẻ phim (Movie Card) */
    .movie-card {
        background-color: #1c2128;
        padding: 15px;
        border-radius: 15px;
        border: 1px solid #30363d;
        text-align: center;
        height: 380px;
        transition: 0.4s;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .movie-card:hover {
        border-color: #58a6ff;
        transform: scale(1.02);
        box-shadow: 0 10px 30px rgba(88, 166, 255, 0.2);
    }
    .movie-title {
        color: #f0f6fc;
        font-size: 1.1rem;
        font-weight: bold;
        height: 55px;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }

    /* Tùy chỉnh Selectbox để dễ nhìn hơn */
    .stSelectbox label { display: none; } /* Ẩn label mặc định để dùng label tùy chỉnh */
    </style>
    """, unsafe_allow_html=True)

# --- 3. Hàm tải dữ liệu ---
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    movies_path = os.path.join(base_path, 'movies.csv')
    if os.path.exists(movies_path):
        return pd.read_csv(movies_path)
    return None

movies = load_data()

if movies is not None:
    # --- 4. SIDEBAR: Tùy chỉnh theo yêu cầu ---
    with st.sidebar:
        st.markdown("<p class='sidebar-title'>🎬 MOVIE MENU</p>", unsafe_allow_html=True)
        st.divider()
        
        # Tiêu đề mới theo yêu cầu
        st.markdown("<span class='sidebar-label'>🔍 Tên phim bạn muốn xem</span>", unsafe_allow_html=True)
        
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
        
        st.markdown("<span class='sidebar-label'>🔢 Số lượng đề xuất</span>", unsafe_allow_html=True)
        num_movies = st.slider("", 4, 24, 12)
        
        st.divider()
        st.info(f"Hệ thống đang lọc các phim thuộc nhóm: {selected_vn}")

    # --- 5. NỘI DUNG CHÍNH ---
    st.markdown(f"<h1 style='text-align: center; color: #58a6ff;'>🍿 KHÁM PHÁ PHIM {selected_vn.upper()}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e; font-size: 1.1rem;'>Danh sách phim được đề xuất dựa trên sở thích cá nhân của bạn</p>", unsafe_allow_html=True)
    st.write("")

    # Lọc phim theo thể loại
    filtered_movies = movies[movies['genres'].str.contains(selected_genre, case=False, na=False)].head(num_movies)

    if not filtered_movies.empty:
        cols = st.columns(4)
        for idx, (_, row) in enumerate(filtered_movies.iterrows()):
            with cols[idx % 4]:
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="https://via.placeholder.com/200x280/161b22/58a6ff?text={selected_vn}" style="width:100%; border-radius:10px;">
                        <div>
                            <div class="movie-title">{row['title']}</div>
                            <p style='color: #8b949e; font-size: 0.85rem; margin-top:5px;'>{row['genres'].replace('|', ' • ')}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                # Nút thông tin
                if st.button("📄 Chi tiết", key=f"info_{row['movieId']}"):
                    st.success(f"Thông tin phim: {row['title']}")
                    st.write(f"Đây là một bộ phim tuyệt vời thuộc thể loại **{selected_vn}**. Bạn có thể tìm xem trên các nền tảng trực tuyến.")
    else:
        st.warning("Không tìm thấy phim thuộc thể loại này.")

else:
    st.error("❌ Không tìm thấy file dữ liệu movies.csv trong thư mục!")
