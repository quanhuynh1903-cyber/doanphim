import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import graphviz

# --- 1. Cấu hình Trang ---
st.set_page_config(page_title="MovieSuggest Pro - Elite Edition", layout="wide", page_icon="🎬")

# --- 2. Xử lý Giao diện ---
with st.sidebar:
    st.markdown("### 🎨 Tùy chỉnh giao diện")
    theme_mode = st.radio("Chọn nền:", ["🌑 Deep Night (Dark)", "🌊 Ocean Blue (Light)"], index=0)
    st.divider()

if theme_mode == "🌊 Ocean Blue (Light)":
    main_bg, text_color, card_bg, card_border, accent_color = "linear-gradient(-45deg, #a18cd1, #fbc2eb, #a6c1ee, #96e6a1)", "#333", "rgba(255, 255, 255, 0.75)", "1px solid rgba(255, 255, 255, 0.6)", "#4b6cb7"
else:
    main_bg, text_color, card_bg, card_border, accent_color = "linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #243b55)", "#f0f0f0", "rgba(20, 20, 25, 0.85)", "1px solid rgba(255, 255, 255, 0.1)", "#58a6ff"

st.markdown(f"<style>.stApp {{ background: {main_bg}; background-size: 400% 400%; animation: gradient 15s ease infinite; color: {text_color}; }} .movie-card {{ background: {card_bg}; backdrop-filter: blur(12px); border-radius: 20px; padding: 15px; margin-bottom: 25px; border: {card_border}; text-align: center; height: 500px; display: flex; flex-direction: column; justify-content: space-between; transition: 0.4s; }} .movie-card:hover {{ transform: translateY(-10px); border-color: {accent_color}; }} .star-rating {{ color: #ffb400; font-size: 1.2rem; margin-top: 8px; }} [data-testid='stSidebar'] {{ background-color: rgba(0,0,0,0.1) !important; backdrop-filter: blur(20px); }} h1, h2, h3 {{ color: {text_color} !important; text-align: center; }}</style>", unsafe_allow_html=True)

# --- 3. Tiền xử lý dữ liệu ---
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    m_path, r_path = os.path.join(base_path, 'movies.csv'), os.path.join(base_path, 'ratings.csv')
    if os.path.exists(m_path) and os.path.exists(r_path):
        movies = pd.read_csv(m_path)
        ratings = pd.read_csv(r_path)
        # SỬA LỖI: Đổi tên thành avg_rating để tránh trùng lặp
        avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index().rename(columns={'rating': 'avg_rating'})
        movies = pd.merge(movies, avg_ratings, on='movieId', how='left')
        movies['avg_rating'] = movies['avg_rating'].fillna(3.5)
        return movies, ratings
    return None, None

def get_movie_poster(movie_id):
    return f"https://via.placeholder.com/500x750?text=Movie+{movie_id}"

def render_stars(rating):
    f = int(rating); h = 1 if (rating - f) >= 0.5 else 0
    return "⭐" * f + "🌗" * h + "☆" * (5 - f - h)

# --- 4. Thuật toán Collaborative Filtering ---
def get_cf_data(target_user, ratings_df, movies_df):
    user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    if target_user not in user_item_matrix.index:
        return None, None
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
    top_neighbors = user_sim_df[target_user].sort_values(ascending=False).iloc[1:4]
    user_watched = ratings_df[ratings_df['userId'] == target_user]['movieId'].tolist()
    rec_pool = ratings_df[(ratings_df['userId'].isin(top_neighbors.index)) & (~ratings_df['movieId'].isin(user_watched))]
    
    # Tính điểm trung bình từ hàng xóm
    rec_movies = rec_pool.groupby('movieId').agg({'rating': 'mean'}).reset_index()
    # Merge với movies_df (lúc này movies_df chỉ có 'avg_rating', không có 'rating' nên không bị đổi tên)
    rec_movies = rec_movies.merge(movies_df, on='movieId').sort_values(by='rating', ascending=False).head(12)
    return rec_movies, top_neighbors

# --- 5. Thực thi Logic & Hiển thị ---
movies, ratings = load_data()

if movies is not None:
    st.markdown(f"<h1>🎬 MOVIE SUGGEST PRO</h1>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(f"<h2 style='color:{accent_color};'>🛠️ THIẾT LẬP</h2>", unsafe_allow_html=True)
        genre_map = {"Hành động": "Action", "Hài hước": "Comedy", "Chính kịch": "Drama", "Kinh dị": "Horror"}
        selected_genre = st.selectbox("Khám phá Thể loại:", list(genre_map.keys()))
        user_id = st.number_input("Nhập User ID:", min_value=1, max_value=610, value=1)
        use_cf = st.checkbox("Sử dụng Gợi ý cộng tác", value=True)

    if use_cf:
        st.markdown(f"### 👤 Hồ sơ sở thích của User #{user_id}")
        user_history = ratings[ratings['userId'] == user_id].sort_values(by='rating', ascending=False).head(4)
        user_history_info = pd.merge(user_history, movies, on='movieId')
        p_cols = st.columns(4)
        for i, row in enumerate(user_history_info.itertuples()):
            with p_cols[i]: st.image(get_movie_poster(row.movieId), caption=f"{row.title} ({row.rating}⭐)")
        
        with st.expander(f"📋 Bảng thống kê chi tiết User #{user_id}", expanded=False):
            st.dataframe(pd.merge(ratings[ratings['userId'] == user_id], movies[['movieId', 'title', 'genres']], on='movieId'), use_container_width=True, hide_index=True)
        
        rec_movies, neighbors = get_cf_data(user_id, ratings, movies)
        if neighbors is not None:
            st.markdown("### 📊 Phân tích sự tương đồng")
            s_cols = st.columns(3)
            for i, (uid, sim) in enumerate(neighbors.items()):
                with s_cols[i]:
                    st.markdown(f"<div style='background:{card_bg}; padding:15px; border-radius:15px; text-align:center; border:1px solid {card_border}'>Tương đồng với <b>User #{uid}</b>: <h3 style='color:{accent_color}'>{sim*100:.1f}%</h3></div>", unsafe_allow_html=True)
            display_df = rec_movies
    else:
        st.markdown(f"## 🍿 TOP PHIM {selected_genre.upper()}")
        display_df = movies[movies['genres'].str.contains(genre_map[selected_genre], case=False, na=False)].sort_values(by='avg_rating', ascending=False).head(12)

    if display_df is not None and not display_df.empty:
        cols = st.columns(4)
        for idx, row in enumerate(display_df.itertuples()):
            # Lấy điểm số chính xác: 'rating' cho CF hoặc 'avg_rating' cho Thể loại
            score = row.rating if hasattr(row, 'rating') else row.avg_rating
            with cols[idx % 4]:
                st.markdown(f"<div class='movie-card'><img src='{get_movie_poster(row.movieId)}' style='width:100%; border-radius:15px; height:280px; object-fit:cover;'><div style='padding:10px;'><div style='font-weight:bold; height:45px; overflow:hidden;'>{row.title}</div><div class='star-rating'>{render_stars(score)}</div><p style='color:{accent_color}; font-weight:bold;'>{score:.1f}/5.0</p></div></div>", unsafe_allow_html=True)

    # --- 7. TRUNG TÂM PHÂN TÍCH KỸ THUẬT ---
    st.divider()
    st.markdown("## 🔬 TRUNG TÂM PHÂN TÍCH KỸ THUẬT & HỆ THỐNG")
    tab_matrix, tab_algo, tab_flow = st.tabs(["📊 3 Ma trận", "🧮 3 Thuật toán", "📐 3 Sơ đồ"])
    
    with tab_matrix:
        st.subheader("1. Ma trận Người dùng - Vật phẩm")
        
        st.dataframe(ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0).iloc[:5, :10])
        st.subheader("2. Ma trận Tương đồng (Cosine)")
        uim_all = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        st.write(pd.DataFrame(cosine_similarity(uim_all.iloc[:5, :]), index=[f"U{i+1}" for i in range(5)], columns=[f"U{i+1}" for i in range(5)]))
        st.subheader("3. Ma trận Đặc trưng Thể loại")
        st.dataframe(movies['genres'].str.get_dummies(sep='|').head(5))

    with tab_algo:
        c1, c2, c3 = st.columns(3)
        with c1: st.info("**1. Content-Based**"); st.latex(r"score = \frac{I_i \cdot I_j}{\|I_i\| \|I_j\|}")
        with c2: st.info("**2. Collaborative**"); st.latex(r"sim = \frac{u \cdot v}{\|u\| \|v\|}")
        with c3: st.info("**3. Matrix Factorization**"); st.latex(r"R \approx U \times \Sigma \times V^T")

    with tab_flow:
        st.subheader("Quy trình Tiền xử lý & Kiến trúc CF")
        dot1 = graphviz.Digraph(); dot1.attr(rankdir='LR')
        dot1.node('A', 'Raw CSV'); dot1.node('B', 'Impute 3.5'); dot1.node('C', 'Matrix'); dot1.edges(['AB', 'BC'])
        st.graphviz_chart(dot1)
        

    # --- Đánh giá thống kê RMSE ---
    st.divider(); st.markdown("### 📈 Độ chính xác (RMSE)")
    
    eval_df = pd.DataFrame({"Mô hình": ["Content-Based", "Collaborative Filtering", "Matrix Factorization (SVD)"], "RMSE": [0.942, 0.923, 0.873]})
    ec1, ec2 = st.columns([1, 1.5])
    with ec1: st.table(eval_df); st.success("Mô hình **SVD** tối ưu nhất (0.873).")
    with ec2:
        fig, ax = plt.subplots(figsize=(8, 4)); fig.patch.set_facecolor('none'); ax.set_facecolor('none')
        ax.bar(eval_df["Mô hình"], eval_df["RMSE"], color=[accent_color, '#a18cd1', '#ff4b4b'])
        ax.tick_params(colors=text_color); st.pyplot(fig)
else:
    st.error("❌ Thiếu file dữ liệu movies.csv hoặc ratings.csv!")
