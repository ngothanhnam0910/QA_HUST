from collections import Counter

# Đường dẫn tới data
path_to_corpus = "/home/gemai/md1/NAMNT_DA/generated_data/corpus.txt"

# Hàm để đếm tần suất xuất hiện của các từ trong tập dữ liệu
def dem_tan_suat_tu(du_lieu):
    words = []
    with open(du_lieu, 'r', encoding='utf-8') as file:
        for line in file:
            # Tách từng từ trong câu và loại bỏ các ký tự đặc biệt
            words.extend([word.lower().strip(".,") for word in line.split()])
    return Counter(words)

# Hàm để tạo bộ từ stop word từ tần suất xuất hiện
def tao_bo_stopwords(tap_du_lieu, so_luong_tu_stopword):
    tan_suat_tu = dem_tan_suat_tu(tap_du_lieu)
    stopwords = [word for word, count in tan_suat_tu.most_common(so_luong_tu_stopword)]
    return stopwords

# Số lượng từ stop word bạn muốn tạo
so_luong_tu_stopword = 200

# Tạo bộ từ stop word từ hai tập dữ liệu
stopwords_data = tao_bo_stopwords(path_to_corpus,so_luong_tu_stopword)


# In ra bộ từ stop word từ hai tập dữ liệu
print("Bộ từ stop words từ tập corpus:")
print(stopwords_data)
