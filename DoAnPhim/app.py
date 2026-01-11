import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Cấu hình Trang ---
st.set_page_config(page_title="Netflix Clone - Recommendation", layout="wide")

# Tùy chỉnh CSS để giống giao diện web phim
st.markdown("""
    <style>
    .main { background-color: #141414; color: white; }
    .stButton>button { width: 100%; border-radius: 5px; background-color: #e50914; color: white; border: none; }
    .movie-card {
        background-color: #2f2f2f;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        height: 300px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Load Dữ liệu ---
@st.cache_data
def load_data():
    # Giả sử bạn dùng tập MovieLens 100k
    movies = pd.read_csv('movies.csv') # Cần có file này
    ratings = pd.read_csv('ratings.csv') # Cần có file này
    return movies, ratings

try:
    movies, ratings = load_data()
    
    # Tiền xử lý ma trận User-Item
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    # --- 2. Xử lý Logic ---
    def get_recommendations(user_id, num=6):
        similar_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:11].index
        similar_ratings = user_item_matrix.loc[similar_users].mean(axis=0)
        user_watched = user_item_matrix.loc[user_id]
        recommendations = similar_ratings[user_watched == 0].sort_values(ascending=False).head(num)
        return movies[movies['movieId'].isin(recommend_recommend_ids := recommendations.index)]

    # --- 3. Giao diện Web ---
    st.title("🍿 MovieFlix")
    
    # Sidebar: Đăng nhập giả lập
    st.sidebar.header("Tài khoản")
    user_id = st.sidebar.number_input("Nhập User ID để cá nhân hóa:", min_value=1, max_value=600, value=1)
    
    st.markdown(f"### Chào mừng trở lại, User #{user_id}!")
    st.write("Dựa trên sở thích của bạn, chúng tôi đề xuất:")

    # Hiển thị kết quả dạng lưới (Grid)
    recommended_movies = get_recommendations(user_id)
    
    cols = st.columns(3) # Tạo 3 cột mỗi dòng
    for idx, (i, row) in enumerate(recommended_movies.iterrows()):
        with cols[idx % 3]:
            # Tạo card cho phim
            st.markdown(f"""
                <div class="movie-card">
                    <img src="https://via.placeholder.com/150x200?text={row['title'][:10]}" style="width:100%; border-radius:5px;">
                    <h4 style="font-size: 14px; margin-top: 10px;">{row['title']}</h4>
                    <p style="color: #b3b3b3; font-size: 12px;">{row['genres']}</p>
                </div>
                """, unsafe_allow_html=True)
            if st.button(f"Xem ngay", key=f"btn_{i}"):
                st.success(f"Đang mở phim: {row['title']}")

    # --- 4. Khám phá thêm ---
    st.divider()
    st.subheader("🔥 Phim đang thịnh hành")
    trending_movies = movies.sample(6)
    t_cols = st.columns(6)
    for idx, (i, row) in enumerate(trending_movies.iterrows()):
        with t_cols[idx]:
            st.image("https://via.placeholder.com/150x220?text=Poster", caption=row['title'][:20])

except Exception as e:
    st.error("Lỗi: Đảm bảo bạn đã có file movies.csv và ratings.csv trong thư mục!")
    st.info("Tải dữ liệu tại: https://grouplens.org/datasets/movielens/latest/")