import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- Cấu hình Trang ---
st.set_page_config(page_title="MovieFlix - Collaborative Filtering", layout="wide")

# Tùy chỉnh CSS để giao diện chuyên nghiệp hơn
st.markdown("""
    <style>
    .main { background-color: #141414 !important; color: white !important; }
    .stApp { background-color: #141414; }
    .movie-card {
        background-color: #2f2f2f;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        height: 380px;
        border: 1px solid #404040;
    }
    h1, h2, h3, p { color: white !important; }
    .stButton>button { 
        width: 100%; border-radius: 5px; 
        background-color: #e50914; color: white; 
        border: none; font-weight: bold;
    }
    .stButton>button:hover { background-color: #ff0a16; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Hàm Load Dữ liệu (Sửa lỗi đường dẫn tại đây) ---
@st.cache_data
def load_movie_data():
    # Lấy đường dẫn tuyệt đối của thư mục chứa file app.py
    base_path = os.path.dirname(__file__)
    
    # Xác định đường dẫn file chính xác dù chạy ở bất cứ đâu
    movies_path = os.path.join(base_path, 'movies.csv')
    ratings_path = os.path.join(base_path, 'ratings.csv')
    
    # Kiểm tra xem file có tồn tại hay không trước khi đọc
    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        return None, None

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings

# Thực thi tải dữ liệu
movies, ratings = load_movie_data()

if movies is not None and ratings is not None:
    # --- 2. Tiền xử lý ma trận User-Item ---
    # Giới hạn dữ liệu nếu cần để tăng tốc độ xử lý (ví dụ: lấy 200 users đầu tiên)
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    # Tính toán độ tương đồng Cosine
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    # --- 3. Logic Khuyến nghị ---
    def get_recommendations(user_id, num=6):
        if user_id not in user_similarity_df.index:
            return pd.DataFrame()
            
        # Lấy top 10 người dùng tương đồng nhất
        similar_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:11].index
        
        # Tính điểm trung bình cộng tác
        similar_ratings = user_item_matrix.loc[similar_users].mean(axis=0)
        
        # Lọc bỏ phim đã xem
        user_watched = user_item_matrix.loc[user_id]
        recommendations = similar_ratings[user_watched == 0].sort_values(ascending=False).head(num)
        
        return movies[movies['movieId'].isin(recommendations.index)]

    # --- 4. Giao diện Người dùng ---
    st.title("🍿 MovieFlix")
    
    st.sidebar.header("🚪 Đăng nhập hệ thống")
    user_id = st.sidebar.number_input("Nhập User ID (1 - 610):", min_value=1, max_value=610, value=1)
    num_rec = st.sidebar.slider("Số lượng phim gợi ý:", 3, 12, 6)
    
    st.markdown(f"### 🎬 Danh sách dành riêng cho User #{user_id}")
    
    rec_movies = get_recommendations(user_id, num_rec)
    
    if not rec_movies.empty:
        # Hiển thị dạng Grid
        cols = st.columns(3)
        for idx, (i, row) in enumerate(rec_movies.iterrows()):
            with cols[idx % 3]:
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="https://via.placeholder.com/180x240?text=🎬" style="width:100%; border-radius:5px; margin-bottom:10px;">
                        <h4 style="height: 50px; overflow: hidden;">{row['title']}</h4>
                        <p style="color: #b3b3b3; font-size: 0.8em;">{row['genres']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                if st.button(f"Phát Phim", key=f"rec_{row['movieId']}"):
                    st.balloons()
                    st.success(f"Đang chuẩn bị phát: {row['title']}")
    else:
        st.warning("Không tìm thấy dữ liệu gợi ý cho người dùng này.")

    # --- Phim thịnh hành ---
    st.divider()
    st.subheader("🔥 Xu hướng hiện nay")
    trending = movies.sample(6)
    t_cols = st.columns(6)
    for idx, (i, row) in enumerate(trending.iterrows()):
        with t_cols[idx]:
            st.image("https://via.placeholder.com/150x220?text=Poster", caption=row['title'][:15])

else:
    # Thông báo lỗi nếu thiếu file
    st.error("❌ Lỗi hệ thống: Không tìm thấy dữ liệu đầu vào!")
    st.info("Vui lòng kiểm tra chắc chắn rằng file **movies.csv** và **ratings.csv** đã được upload lên GitHub trong cùng thư mục với app.py.")
    st.markdown("[Tải dữ liệu tại đây](https://grouplens.org/datasets/movielens/latest/) (Chọn bản ml-latest-small.zip)")
