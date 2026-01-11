import streamlit as st
import pandas as pd
import os

# --- 1. Cấu hình Trang ---
st.set_page_config(page_title="MovieSuggest Pro", layout="wide", page_icon="🎬")

# --- 2. CUSTOM CSS: Sidebar nổi bật và ẩn các thành phần thừa ---
st.markdown("""
    <style>
    /* Nền tối cho trang */
    .stApp { background-color: #0d1117; }
    
    /* Sidebar nổi bật */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 3px solid #58a6ff;
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
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .sidebar-label {
        color: #ffffff !important;
        font-size: 1.4rem !important; /* Tăng kích cỡ chữ */
        font-weight: 700 !important;
        margin-top: 30px;
        margin-bottom: 10px;
        display: block;
    }

    /* Thẻ phim (Movie Card) tối giản */
    .movie-card {
        background-color: #1c2128;
        padding: 15px;
        border-radius: 15px;
        border: 1px solid #30363d;
        text-align: center;
        height: 330px; /* Thu gọn vì đã bỏ nút */
        transition: 0.4s;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin-bottom: 20px;
    }
    .movie-card:hover {
        border-color: #58a6ff;
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(88, 166, 255, 0.2);
    }
    .movie-title {
        color: #f0f6fc;
        font-size: 1.15rem;
        font-weight: bold;
        margin-top: 15px;
        height: 60px;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    
    /* Ẩn label mặc định của Streamlit để dùng custom label */
    .stSelectbox label, .stSlider label { display: none; }
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
    # --- 4. SIDEBAR ---
    with st.sidebar:
        st.markdown("<p class='sidebar-title'>🎬 MOVIE MENU</p>", unsafe_allow_html=True)
        st.divider()
        
        # Tiêu đề mới theo yêu cầu: "Dạng phim bạn muốn xem"
        st.markdown("<span class='sidebar-label'>🔍 Dạng phim bạn muốn xem</span>", unsafe_allow_html=True)
        
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
        num_movies = st.slider("Số lượng", 4, 24, 12)
        
        st.divider()
        st.info(f"Đang tìm kiếm phim: {selected_vn}")

    # --- 5. NỘI DUNG CHÍNH ---
    st.markdown(f"<h1 style='text-align: center; color: #58a6ff;'>🍿 ĐỀ XUẤT PHIM {selected_vn.upper()}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e; font-size: 1.1rem;'>Danh sách phim được chọn lọc riêng cho bạn</p>", unsafe_allow_html=True)
    st.write("")

    # Lọc phim
    filtered_movies = movies[movies['genres'].str.contains(selected_genre, case=False, na=False)].head(num_movies)

    if not filtered_movies.empty:
        cols = st.columns(4)
        for idx, (_, row) in enumerate(filtered_movies.iterrows()):
            with cols[idx % 4]:
                # Chỉ hiển thị poster và tên phim, không có nút bấm
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="https://via.placeholder.com/200x280/161b22/58a6ff?text={selected_vn}" style="width:100%; border-radius:10px;">
                        <div class="movie-title">{row['title']}</div>
                        <p style='color: #8b949e; font-size: 0.85rem; margin-top:5px;'>{row['genres'].split('|')[0]}</p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Không tìm thấy dữ liệu cho thể loại này.")

else:
    st.error("❌ Không tìm thấy file movies.csv!")
