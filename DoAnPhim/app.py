import streamlit as st
import pandas as pd
import os

# --- Cấu hình Trang ---
st.set_page_config(page_title="MovieSuggest - Đề Xuất Phim", layout="wide", page_icon="🎬")

# --- CUSTOM CSS: Làm nổi bật Sidebar và chữ ---
st.markdown("""
    <style>
    /* Nền tối cho toàn bộ trang */
    .stApp { background-color: #0e1117; }
    
    /* Làm nổi bật Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1c23 !important;
        border-right: 2px solid #e50914; /* Viền đỏ làm điểm nhấn */
        padding-top: 20px;
    }

    /* Tùy chỉnh chữ trong Sidebar cho rõ ràng */
    [data-testid="stSidebar"] .stMarkdown h1, 
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown p {
        color: #ffffff !important;
        font-weight: bold !important;
    }
    
    /* Làm nổi bật nhãn của Selectbox */
    .stSelectbox label p {
        font-size: 1.2rem !important;
        color: #ff4b4b !important;
        font-weight: bold !important;
    }

    /* Thẻ phim (Movie Card) */
    .movie-card {
        background-color: #1c2128;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #30363d;
        text-align: center;
        height: 320px;
        transition: 0.3s;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .movie-card:hover {
        border-color: #e50914;
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.3);
    }
    .movie-title {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: bold;
        margin-top: 10px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    h1 { color: #e50914 !important; font-size: 2.5rem !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Tải dữ liệu ---
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    movies_path = os.path.join(base_path, 'movies.csv')
    if not os.path.exists(movies_path):
        return None
    return pd.read_csv(movies_path)

movies = load_data()

if movies is not None:
    # --- SIDEBAR: Phần chọn phim/thể loại ---
    with st.sidebar:
        st.markdown("<h1 style='text-align: center;'>🎬</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>DAN
