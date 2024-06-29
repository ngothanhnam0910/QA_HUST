import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from rank_bm25 import *
from utils_chunking import bm25_tokenizer, load_json
import argparse

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_pair", default=20, type=int)
    parser.add_argument("--model_path", default="/home/gemai/md1/NAMNT_DA2/save_bm25/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--data_path", default="./data", type=str, help="path to input data")
    parser.add_argument("--save_pair_path", default="/home/gemai/md1/NAMNT_DA2/data_train_sbert/pair_data/", type=str, help="path to save pair sentence directory")
    args = parser.parse_args()

    train_path = os.path.join(args.data_path, "data_qa.json")
    training_items = load_json(train_path)["items"]

    with open(args.model_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    with open("/home/gemai/md1/NAMNT_DA2/save_bm25/doc_refers_saved", "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)

    doc_data = json.load(open(os.path.join("/home/gemai/md1/NAMNT_DA2/generated_data", "legal_dict.json")))

    save_pairs = []
    top_n = args.top_pair
    for idx, item in tqdm(enumerate(training_items)):
        question_id = item["question_id"]
        question = item["question"]
        relevant_articles = item["relevant_info"]
        actual_positive = len(relevant_articles)
        
        tokenized_query = bm25_tokenizer(question)
        doc_scores = bm25.get_scores(tokenized_query)

        predictions = np.argpartition(doc_scores, len(doc_scores) - top_n)[-top_n:]

        # Save positive pairs
        for article in relevant_articles:

            concat_id = article["Field_id"].strip() + "_" + article["infor_id"].strip()
            # chunking passage
            text_chunking = chunk_string(doc_data[concat_id]["text"])
            for chunk in text_chunking:
                save_dict = {}
                save_dict["question"] = question
                save_dict["document"] = doc_data[concat_id]["title"] + " " + chunk
                save_dict["relevant"] = 1
                save_pairs.append(save_dict)


        # Save negative pairs
        for idx, idx_pred in enumerate(predictions):
            pred = doc_refers[idx_pred]

            check = 0
            for article in relevant_articles:
                if pred[0] == article["Field_id"] and pred[1] == article["infor_id"]:
                    check += 1
 
            if check == 0:
                # print(f"Nhay vao check bang 0 line 64s")
                concat_id = pred[0] + "_" + pred[1]
                # chunking passage
                text_chunking = chunk_string(doc_data[concat_id]["text"])
                for chunk in text_chunking:
                    save_dict = {}
                    save_dict["question"] = question
                    save_dict["document"] = doc_data[concat_id]["title"] + " " + chunk
                    save_dict["relevant"] = 0
                    save_pairs.append(save_dict)

    #save_path = args.save_pair_path
    save_path = f"/home/gemai/md1/NAMNT_DA2/data_train_sbert/pair_data"
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"bm_25_pairs_top{top_n}_chunking"), "wb") as pair_file:
        pickle.dump(save_pairs, pair_file)
        print(f"dump sucessful")
    print(len(save_pairs))