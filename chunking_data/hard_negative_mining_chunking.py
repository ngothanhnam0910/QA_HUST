import pickle
import os
import numpy as np
import json
import torch
from tqdm import tqdm
from rank_bm25 import *
import argparse
import warnings 
from sentence_transformers import SentenceTransformer, util
warnings.filterwarnings("ignore")

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
    parser.add_argument("--model_path", default="/home/namnt/md1/NAMNT_DA2/save_bm25/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--sentence_bert_path", default="/home/namnt/md1/sources/namnt/checkpoint/ckp_round1_cocondenser_ver2", type=str, help="path to round 1 sentence bert model")
    parser.add_argument("--data_path", default="./data", type=str, help="path to input data")
    parser.add_argument("--save_path", default="/home/namnt/md1/NAMNT_DA2/data_train_sbert/pair_data", type=str)
    parser.add_argument("--top_k", default=20, type=str, help="top k hard negative mining")
    parser.add_argument("--path_doc_refer", default="/home/namnt/md1/NAMNT_DA2/save_bm25/doc_refers_saved_chunking", type=str, help="path to doc refers")
    parser.add_argument("--path_legal", default="/home/namnt/md1/NAMNT_DA2/generated_data/legal_dict_chunking.json", type=str, help="path to legal dict")
    parser.add_argument("--path_save_load_embedding_doc",default ="/home/namnt/md1/NAMNT_DA2/generated_data/legal_corpus_cocondenser_embedding_chunking_ver2.pkl",type=str, help="path to embedding title and text of document")
    args = parser.parse_args()

    # load training data from json
    data = json.load(open(os.path.join(args.data_path, "data_qa.json")))

    training_data = data["items"]
    print(len(training_data))

    # load bm25 model
    with open(args.model_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)

    with open(args.path_doc_refer, "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)

    doc_path = os.path.join(args.path_legal)
    df = open(doc_path)
    doc_data = json.load(df)
    
    # load hard negative model
    model = SentenceTransformer(args.sentence_bert_path)

    # add embedding for data
    # if you already have data with encoded sentence uncoment line 47 - 54
    import pickle
    embed_list = []
    for k, v in tqdm(doc_data.items()):
        doc_data[k]['embedding'] = []
        for text in v['text']:
            embed = model.encode(v['title'] + ' ' + text)
            doc_data[k]['embedding'].append(embed)

    with open(args.path_save_load_embedding_doc, 'wb') as pkl:
        pickle.dump(doc_data, pkl)

    with open(args.path_save_load_embedding_doc, 'rb') as pkl:
        data = pickle.load(pkl)

    pred_list = []
    top_k = args.top_k
    save_pairs = []

    for idx, item in tqdm(enumerate(training_data)):
        question_id = item["question_id"]
        question = item["question"]
        relevant_articles = item["relevant_info"]
        actual_positive = len(relevant_articles)
        
        for article in relevant_articles:
            concat_id = article["Field_id"].strip() + "_" + article["infor_id"].strip()
            # chunking passage
            # text_chunking = chunk_string(doc_data[concat_id]["text"])
            for chunk in doc_data[concat_id]["text"]:
                save_dict = {}
                save_dict["question"] = question
                save_dict["document"] = doc_data[concat_id]["title"] + " " + chunk
                save_dict["relevant"] = 1
                save_pairs.append(save_dict)

        # embedding question
        encoded_question  = model.encode(question)
        list_embs = []

        # embedding title + text của phần điều luật
        for k, v in data.items():
            for emb in v['embedding']:
                emb_2 = torch.tensor(emb).unsqueeze(0)
                list_embs.append(emb_2)

        matrix_emb = torch.cat(list_embs, dim=0)

        # Tính độ đo cosin giữa embedding của các điều luat và embedding của câu hỏi => chọn topK score cao nhất.
        all_cosine = util.cos_sim(encoded_question, matrix_emb).numpy().squeeze(0)
        predictions = np.argpartition(all_cosine, len(all_cosine) - 20)[-20:]
        
        
        for idx, idx_pred in enumerate(predictions):

            # lấy ra phần nôi dung có chỉ số bằng với trong predictions
            pred = doc_refers[idx_pred]
                
            check = 0
            for article in relevant_articles:
                # Nếu thông tin predict trùng với ground truth thì check tăng 1
                check += 1 if pred[0] == article["Field_id"] and pred[1] == article["infor_id"] else 0

            if check == 0:
                # Tạo ra các cặp negative từ dự doan sai cua model
                save_dict = {}
                save_dict["question"] = question
                # concat_id = pred[0] + "_" + pred[1]
                save_dict["document"] = pred[2]
                save_dict["relevant"] = 0
                save_pairs.append(save_dict)
    print(f"save pairs:{len(save_pairs)}")
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"save_pairs_cocondenser_top{top_k}_chunking_ver2.pkl"), "wb") as pair_file:
        pickle.dump(save_pairs, pair_file)
