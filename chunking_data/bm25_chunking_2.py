import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from rank_bm25 import *
import argparse
from utils_chunking import bm25_tokenizer, calculate_f2
# from config import Config
import underthesea

def chunk_string(input_string):
    # Tách văn bản thành các câu
    sentences = underthesea.sent_tokenize(input_string)
    new_sentences = []
    for i in range(len(sentences)):
        if len(sentences[i]) > 2:
            if len(sentences[i - 1]) > 2:
                new_sentences.append(sentences[i])
            else:
                new_sentences.append(sentences[i - 1] + sentences[i])

    new_sentences = sentences
    chunks = []
    current_chunk = ""

    # Duyệt qua từng câu
    for sentence in sentences:
        # Nếu thêm câu vào chunk hiện tại không làm cho chunk vượt quá 200 từ
        if len(current_chunk.split()) + len(sentence.split()) <= 200:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
class Config:
    data_path = "./data"
    save_bm25 = "/home/gemai/md1/NAMNT_DA2/save_bm25"
    top_k_bm25 = 2
    bm25_k1 = 0.4
    bm25_b = 0.6

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # load document to save running time, 
    # must run 1 time if we change pre-process step
    parser.add_argument("--load_docs", action="store_false")
    parser.add_argument("--num_eval", default=1000, type=str)
    args = parser.parse_args()
    cfg = Config()
    
    save_path = cfg.save_bm25
    os.makedirs(save_path, exist_ok=True)

    raw_data = cfg.data_path
    corpus_path = os.path.join(raw_data, "template.json")

    data = json.load(open(corpus_path))

    if args.load_docs:
        print("Process documents")
        documents = []
        doc_refers = []
        for law_article in tqdm(data):
            law_id = law_article["Field_id"]
            law_articles = law_article["infor"]
            
            for sub_article in law_articles:
                article_id = sub_article["infor_id"]
                article_title = sub_article["title"]
                article_text = sub_article["text"]

                # chunking text
                text_chunks = chunk_string(article_text)
                for chunk in text_chunks:
                    article_full = article_title + " " + chunk
                    tokens = bm25_tokenizer(article_full)
                    documents.append(tokens)
                    doc_refers.append([law_id, article_id, article_full])
        
        with open(os.path.join(save_path, "documents_manual_chunking"), "wb") as documents_file:
            pickle.dump(documents, documents_file)
        with open(os.path.join(save_path,"doc_refers_saved_chunking"), "wb") as doc_refer_file:
            pickle.dump(doc_refers, doc_refer_file)
    # else:
    #     with open(os.path.join(save_path, "documents_manual"), "rb") as documents_file:
    #         documents = pickle.load(documents_file)
    #     with open(os.path.join(save_path,"doc_refers_saved"), "rb") as doc_refer_file:
    #         doc_refers = pickle.load(doc_refer_file)
            
