import pandas as pd
import requests
import os
import time

# --- 1. Cấu hình API TMDB ---
# QUAN TRỌNG: Hãy dán API Key (v3 auth) của bạn vào đây
TMDB_API_KEY = 'YOUR_API_KEY_CỦA_BẠN' 

# Thư mục lưu trữ ảnh (Phải khớp với LOCAL_POSTER_DIR trong app.py)
POSTER_DIR = "local_posters"

if not os.path.exists(POSTER_DIR):
    os.makedirs(POSTER_DIR)

def download_poster(movie_id, movie_title):
    file_name = f"{POSTER_DIR}/{movie_id}.jpg"
    
    # Kiểm tra nếu đã có ảnh rồi thì bỏ qua không tải lại
    if os.path.exists(file_name):
        return True

    try:
        # Làm sạch tên phim để tìm kiếm chính xác hơn
        search_title = movie_title.split(' (')[0]
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={search_title}"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if response.status_code == 200 and data.get('results'):
            # Lấy poster của kết quả tìm kiếm đầu tiên
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                img_data = requests.get(poster_url, timeout=10).content
                with open(file_name, 'wb') as f:
                    f.write(img_data)
                print(f"✅ Đã tải: {movie_title}")
                return True
        else:
            print(f"⚠️ Không tìm thấy poster cho: {movie_title}")
    except Exception as e:
        print(f"❌ Lỗi khi xử lý phim {movie_title}: {e}")
    
    return False

if __name__ == "__main__":
    print("🚀 Bắt đầu quá trình thu thập poster phim...")

    # Kiểm tra file dữ liệu
    if not os.path.exists('movies.csv'):
        print("❌ Lỗi: Không tìm thấy file 'movies.csv'.")
    else:
        movies_df = pd.read_csv('movies.csv')
        
        # Số lượng phim muốn có ảnh
        target_count = 100 
        
        # Ưu tiên lấy những phim có đánh giá cao nếu có file ratings
        if os.path.exists('ratings.csv'):
            ratings_df = pd.read_csv('ratings.csv')
            avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
            movies_df = pd.merge(movies_df, avg_ratings, on='movieId', how='left')
            # Sắp xếp giảm dần theo rating và chỉ lấy những phim có lượt đánh giá đủ tốt
            movies_to_process = movies_df.sort_values(by='rating', ascending=False).head(target_count * 2)
        else:
            movies_to_process = movies_df.head(target_count * 2)

        success_count = 0
        for _, row in movies_to_process.iterrows():
            if success_count >= target_count:
                break
                
            if download_poster(row['movieId'], row['title']):
                success_count += 1
            
            # Tránh bị TMDB chặn do gửi yêu cầu quá nhanh
            time.sleep(0.2)

        print(f"\n✨ HOÀN TẤT! Đã có {success_count} ảnh trong thư mục '{POSTER_DIR}'.")
        print("Bây giờ bạn có thể nén thư mục này lại cùng với app.py để mang đi thuyết trình.")
