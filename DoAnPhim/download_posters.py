import pandas as pd
import requests
import os
import time

# --- Cấu hình API TMDB ---
# THAY THẾ 'YOUR_API_KEY_CỦA_BẠN' BẰNG API KEY THẬT CỦA BẠN
TMDB_API_KEY = 'YOUR_API_KEY_CỦA_BẠN' 

# Đường dẫn thư mục chứa poster
POSTER_DIR = "local_posters"
if not os.path.exists(POSTER_DIR):
    os.makedirs(POSTER_DIR)

# --- Hàm tìm và tải poster ---
def download_poster_for_movie(movie_id, movie_title):
    try:
        search_title = movie_title.split(' (')[0] # Loại bỏ năm trong tiêu đề
        
        # Gọi API để tìm kiếm phim
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={search_title}&language=en-US"
        response = requests.get(url).json()
        
        if response and response['results']:
            # Lấy kết quả đầu tiên (thường là kết quả tốt nhất)
            result = response['results'][0]
            if result['poster_path']:
                poster_url = f"https://image.tmdb.org/t/p/w500{result['poster_path']}"
                file_name = f"{POSTER_DIR}/{movie_id}.jpg"
                
                # Tải ảnh về
                img_data = requests.get(poster_url).content
                with open(file_name, 'wb') as handler:
                    handler.write(img_data)
                print(f"✅ Tải thành công poster cho {movie_title} (ID: {movie_id})")
                return True
    except Exception as e:
        print(f"❌ Lỗi khi tải poster cho {movie_title} (ID: {movie_id}): {e}")
    return False

# --- Main script ---
if __name__ == "__main__":
    print("Bắt đầu tải poster phim...")
    
    # Đảm bảo file movies.csv nằm cùng thư mục
    if not os.path.exists('movies.csv'):
        print("Lỗi: Không tìm thấy file 'movies.csv'. Vui lòng đặt nó cùng thư mục với script này.")
    else:
        movies = pd.read_csv('movies.csv')
        
        # Sắp xếp phim theo độ phổ biến (giả định) và lấy TOP N
        # Bạn có thể điều chỉnh số lượng phim muốn tải tại đây
        num_movies_to_download = 100 
        
        # Nếu có file ratings.csv, chúng ta có thể sắp xếp theo rating để tải các phim "hay"
        if os.path.exists('ratings.csv'):
            ratings = pd.read_csv('ratings.csv')
            avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
            movies = pd.merge(movies, avg_ratings, on='movieId', how='left')
            movies_to_process = movies.sort_values(by='rating', ascending=False).head(num_movies_to_download)
        else:
            movies_to_process = movies.head(num_movies_to_download) # Lấy 100 phim đầu tiên nếu không có ratings
            
        downloaded_count = 0
        for index, row in movies_to_process.iterrows():
            if download_poster_for_movie(row['movieId'], row['title']):
                downloaded_count += 1
            time.sleep(0.1) # Dừng một chút để tránh bị API chặn (rate limiting)
            
        print(f"\nHoàn tất! Đã tải về {downloaded_count} poster vào thư mục '{POSTER_DIR}'.")
