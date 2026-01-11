import pandas as pd
import requests
import os
import time

# --- CẤU HÌNH ---
# Hãy đảm bảo bạn lấy đúng API Key (v3 auth) từ themoviedb.org
TMDB_API_KEY = 'DÁN_API_KEY_THẬT_CỦA_BẠN_VÀO_ĐÂY' 
POSTER_DIR = "local_posters"

if not os.path.exists(POSTER_DIR):
    os.makedirs(POSTER_DIR)

def download_poster(movie_id, movie_title):
    file_name = f"{POSTER_DIR}/{movie_id}.jpg"
    if os.path.exists(file_name): return True

    try:
        search_title = movie_title.split(' (')[0]
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={search_title}"
        response = requests.get(url, timeout=10)
        
        # Kiểm tra mã phản hồi từ server
        if response.status_code == 401:
            print("❌ Lỗi: API Key không hợp lệ hoặc chưa được kích hoạt!")
            return False
            
        data = response.json()
        if data.get('results'):
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                img_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                img_data = requests.get(img_url).content
                with open(file_name, 'wb') as f:
                    f.write(img_data)
                print(f"✅ Đã tải: {movie_title}")
                return True
        print(f"⚠️ Không tìm thấy ảnh cho: {movie_title}")
    except Exception as e:
        print(f"❌ Lỗi kết nối: {e}")
    return False

if __name__ == "__main__":
    if not os.path.exists('movies.csv'):
        print("❌ Thiếu file movies.csv!")
    else:
        df = pd.read_csv('movies.csv').head(100) # Tải trước 100 phim
        for _, row in df.iterrows():
            download_poster(row['movieId'], row['title'])
            time.sleep(0.2)
        print("\n✨ Xong! Hãy kiểm tra thư mục local_posters.")
