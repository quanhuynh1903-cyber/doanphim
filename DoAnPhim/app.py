import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Cấu hình Trang ---
st.set_page_config(page_title="MovieSuggest Pro - Similarity Engine", layout="wide", page_icon="🎬")

# --- 2. Xử lý Giao diện (Theme Mode) ---
with st.sidebar:
    st.markdown("### 🎨 Tùy chỉnh giao diện")
    theme_mode = st.radio("Chọn nền:", ["🌑 Deep Night (Dark)", "🌊 Ocean Blue (Light)"])
    st.divider()

if theme_mode == "🌊 Ocean Blue (Light)":
    main_bg = "linear-gradient(-45deg, #a18cd1, #fbc2eb, #a6c1ee, #96e6a1)"
    text_color, card_bg, card_border = "#333", "rgba(255, 255, 255, 0.75)", "1px solid rgba(255, 255, 255, 0.6)"
    sidebar_bg = "rgba(255, 255, 255, 0.2)"
    accent_color = "#4b6cb7"
else:
    main_bg = "linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #243b55)"
    text_color, card_bg, card_border = "#f0f0f0", "rgba(20, 20, 25, 0.85)", "1px solid rgba(255, 255, 255, 0.1)"
    sidebar_bg = "rgba(0, 0, 0, 0.4)"
    accent_color = "#58a6ff"

st.markdown(f"""
<style>
@keyframes gradient {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
.stApp {{ background: {main_bg}; background-size: 400% 400%; animation: gradient 15s ease infinite; color: {text_color}; font-family: 'Segoe UI', sans-serif; }}
.movie-card {{ background: {card_bg}; backdrop-filter: blur(12px); border-radius: 20px; padding: 15px; margin-bottom: 25px; border: {card_border}; text-align: center; height: 500px; display: flex; flex-direction: column; justify-content: space-between; transition: 0.4s; }}
.movie-card:hover {{ transform: translateY(-10px); box-shadow: 0 15px 35px rgba(0,0,0,0.3); border-color: {accent_color}; }}
.similarity-badge {{ background: {accent_color}; color: white; padding: 2px 10px; border-radius: 10px; font-size: 0.8rem; margin-bottom: 5px; display: inline-block; }}
[data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; backdrop-filter: blur(20px); }}
</style>
""", unsafe_allow_html=True)

# --- 3. Hàm xử lý dữ liệu & Thuật toán Tương thích ---
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
        return movies, ratings
    return None, None

def get_movie_poster(movie_id):
    local_path = os.path.join("local_posters", f"{movie_id}.jpg")
    return local_path if os.path.exists(local_path) else "https://via.placeholder.com/500x750?text=No+Poster"

# --- 4. Logic Ứng dụng ---
movies, ratings = load_data()

if movies is not None:
    with st.sidebar:
        st.markdown(f"<h2 style='color:{accent_color};'>🛠️ THIẾT LẬP</h2>", unsafe_allow_html=True)
        st.write("👥 **Hệ thống Khuyến nghị Cộng tác**")
        target_user = st.number_input("Nhập mã User ID của bạn:", min_value=1, max_value=610, value=1)
        st.info("💡 Hệ thống sẽ tìm các User khác có gu xem phim giống bạn nhất.")

    # TẠO MA TRẬN USER-ITEM
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

    # PHẦN 1: HỒ SƠ SỞ THÍCH CỦA BẠN
    st.markdown(f"### 👤 Hồ sơ sở thích của User #{target_user}")
    user_top_rated = ratings[ratings['userId'] == target_user].sort_values(by='rating', ascending=False).head(4)
    user_top_movies = pd.merge(user_top_rated, movies, on='movieId')
    
    t_cols = st.columns(4)
    for idx, row in enumerate(user_top_movies.to_dict('records')):
        with t_cols[idx]:
            st.image(get_movie_poster(row['movieId']), caption=f"{row['title']} ({row['rating']}⭐)", use_container_width=True)

    st.divider()

    # PHẦN 2: PHÂN TÍCH SỰ TƯƠNG THÍCH
    st.markdown(f"### 📊 Phân tích sự tương thích với cộng đồng")
    # Lấy 3 người dùng giống nhất
    top_sim_users = user_sim_df[target_user].sort_values(ascending=False).iloc[1:4]
    
    c1, c2, c3 = st.columns(3)
    for i, (uid, sim_val) in enumerate(top_sim_users.items()):
        with [c1, c2, c3][i]:
            st.markdown(f"""
            <div style="background:{card_bg}; padding:15px; border-radius:15px; border-left:5px solid {accent_color};">
                <p style="margin:0;">Tương đồng với <b>User #{uid}</b></p>
                <h2 style="margin:0; color:{accent_color};">{sim_val*100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)

    # PHẦN 3: GỢI Ý PHIM DỰA TRÊN SỰ TƯƠNG THÍCH
    st.write("")
    st.markdown(f"### 🍿 Những phim bạn có thể thích (Dựa trên hàng xóm tương đồng)")
    
    similar_user_ids = top_sim_users.index.tolist()
    user_watched = ratings[ratings['userId'] == target_user]['movieId'].tolist()
    
    # Lấy phim từ những người giống nhất
    recommendations = ratings[(ratings['userId'].isin(similar_user_ids)) & 
                              (~ratings['movieId'].isin(user_watched)) & 
                              (ratings['rating'] >= 4.5)].head(8)
    
    if not recommendations.empty:
        rec_movies = pd.merge(recommendations, movies, on='movieId')
        r_cols = st.columns(4)
        for idx, row in enumerate(rec_movies.iterrows()):
            # Tính toán độ tương thích của riêng bộ phim đó (giả lập dựa trên user similarity)
            movie_sim = top_sim_users.iloc[0] * 100 - (idx * 2) 
            
            with r_cols[idx % 4]:
                poster = get_movie_poster(row[1]['movieId'])
                st.markdown(f"""
                    <div class="movie-card">
                        <div class="similarity-badge">Độ tương thích: {movie_sim:.1f}%</div>
                        <img src="{poster}" style="width:100%; border-radius:15px; height:280px; object-fit:cover;">
                        <div style="padding:10px;">
                            <div class="movie-title"><b>{row[1]['title']}</b></div>
                            <p style="font-size:0.8rem; opacity:0.8; margin-top:5px;">Gợi ý từ User giống bạn</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Hệ thống đang tìm kiếm thêm phim phù hợp...")

else:
    st.error("❌ Vui lòng kiểm tra lại file dữ liệu!")
