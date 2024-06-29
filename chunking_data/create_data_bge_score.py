import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from rank_bm25 import *
from utils import bm25_tokenizer, load_json
from sentence_transformers import SentenceTransformer, util
import argparse
from FlagEmbedding import FlagReranker
import math

def sigmoid(x):
    return 1/(1 + math.exp(-x))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_pair", default=20, type=int)
    parser.add_argument("--model_path",
                        default="/home/ubuntu/namnt/DATN/model/bm25_Plus_04_06_model_full_manual_stopword",
                        type=str)
    parser.add_argument("--data_path", default="data", type=str, help="path to input data")
    parser.add_argument("--save_pair_path", default="/home/ubuntu/namnt/FlagEmbedding/data", type=str,help="path to save pair sentence directory")
    args = parser.parse_args()

    print(f"Starting load rerannker model")
    reranker = FlagReranker('/home/ubuntu/namnt/FlagEmbedding/checkpoint/reranking/checkpoint-5000', use_fp16=True)

    # Load data
    train_path = os.path.join(args.data_path, "data_qa.json")
    training_items = load_json(train_path)["items"]

    with open(args.model_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    with open("data/doc_refers_saved", "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)

    doc_data = json.load(open("data/legal_dict.json"))
    top_n = args.top_pair

    file_name = "data_finetune_bge_round1_distillation"
    #file_name = "data_for_reranking"
    #file_name = "data_distillation"
    with open(os.path.join(args.save_pair_path, file_name + '.json'), 'w') as output_file:
        for idx, item in tqdm(enumerate(training_items)):
            question_id = item["question_id"]
            question = item["question"]
            relevant_articles = item["relevant_info"]
            actual_positive = len(relevant_articles)

            tokenized_query = bm25_tokenizer(question)
            doc_scores = bm25.get_scores(tokenized_query)

            predictions = np.argpartition(doc_scores, len(doc_scores) - top_n)[-top_n:]

            # create_data for bge
            save_dict = {}
            save_dict["query"] = question
            save_dict["pos"] = []
            save_dict["neg"] = []
            save_dict["pos_scores"] = []
            save_dict["neg_scores"] = []

            # embedding question
            # question_emb = model.encode(question)
            #save_dict["prompt"] = "Cho trước một câu truy vấn A và một đoạn văn B, xác định xem đoạn văn có chứa câu trả lời cho câu truy vấn không bằng cách cung cấp dự đoán là Yes' hoặc 'No'."
            for article in relevant_articles:
                concat_id = article["Field_id"].strip() + "_" + article["infor_id"].strip()
                save_dict["pos"].append(doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"])
                scores = reranker.compute_score([(doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"]), question])
                save_dict["pos_scores"].append(sigmoid(scores))

            # Save negative pairs
            for idx, idx_pred in enumerate(predictions):
                pred = doc_refers[idx_pred]

                check = 0
                for article in relevant_articles:
                    if pred[0] == article["Field_id"] and pred[1] == article["infor_id"]:
                        check += 1

                if check == 0: # không trùng với ground truth thì cho làm negative
                    concat_id = pred[0] + "_" + pred[1]
                    save_dict["neg"].append(doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"])
                    scores = reranker.compute_score([(doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"]), question])
                    save_dict["neg_scores"].append(sigmoid(scores))

            # lưu save_dict vao file output
            output_file.write(json.dumps(save_dict,ensure_ascii=False) + '\n')
            print(f"write sucessful")

    # save_path = args.save_pair_path
    # os.makedirs(save_path, exist_ok=True)
    # with open(os.path.join(save_path, f"bm_25_pairs_top{top_n}"), "wb") as pair_file:
    #     pickle.dump(save_pairs, pair_file)
    #     print(f"dump sucessful")
    # print(len(save_pairs))