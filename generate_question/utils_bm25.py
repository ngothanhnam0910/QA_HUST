import string
from underthesea import word_tokenize
import os
import json

#number = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
#chars = ["a", "b", "c", "d", "đ", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"]
stop_word = ["của", "và", "các", "có", "được", "theo", "tại", "trong", "về",
             "người",  "này", "khoản", "cho", "không", "từ", "phải",
            "ngày", "việc", "sau",  "để",  "đến", "bộ",  "với", "là", "năm",
            "khi", "số", "trên", "khác", "đã", "thì", "thuộc", "đồng",
            "do", "một", "bị", "vào", "lại", "ở", "nếu", "làm", "đây",
            "như", "đó", "mà", "nơi", "”", "“", "những", "tập"]

def remove_stopword(w):
    return w not in stop_word
def remove_punctuation(w):
    return w not in string.punctuation
def lower_case(w):
    return w.lower()

def bm25_tokenizer(text):
    tokens = word_tokenize(text)
    tokens = list(map(lower_case, tokens))
    tokens = list(filter(remove_punctuation, tokens))
    tokens = list(filter(remove_stopword, tokens))
    return tokens

def calculate_f2(precision, recall):
    return (5 * precision * recall) / (4 * precision + recall + 1e-20)

def load_json(path):
    return json.load(open(path))

# if __name__ == "__main__":
#     text = "Năm học 2023-2024 Nhà trường dành khoảng 70 tỷ đồng làm quỹ học bổng KKHT (khuyến khích học tập) cho những sinh viên có kết quả học tập và rèn luyện tốt."
#     list_token = bm25_tokenizer(text)
#     print(f"text: {text}")
#     print(f"list token: {list_token}")