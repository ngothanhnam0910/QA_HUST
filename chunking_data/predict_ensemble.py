import numpy as np
import json
import torch
from tqdm import tqdm
from rank_bm25 import *
import argparse
import os
import pickle
import glob
from utils_chunking import bm25_tokenizer
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder
from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding import FlagReranker
import math
import underthesea


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

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


def encode_legal_data(legal_dict_json, model):
    # Load phần document liên quan đến các điều luật
    doc_data = json.load(open(legal_dict_json))
    emb2_list = []
    for k, doc in tqdm(doc_data.items()):
        text_chunks = chunk_string(doc_data[k]["text"])
        emb_chunk = []
        for chunk in text_chunks:
            text_for_emb = doc_data[k]["title"] + " " + chunk
            emb_chunk.append(model.encode(text_for_emb))
        emb2_list.append(emb_chunk)
    return emb2_list


def get_reranking_score(question, text_legal_data, model):
    list_reranking_score = model.compute_score([[question, item] for item in text_legal_data])
    return list_reranking_score

def encode_question(question_data, model):
    print("Start encoding questions.")
    question_embs = []

    for _, item in tqdm(enumerate(question_data)):
        question = item["question"]
        emb_quest = model.encode(question)
        question_embs.append(emb_quest)
    return question_embs

def load_encoded_legal_corpus(legal_data_path):
    print("Start loading legal corpus.")
    with open(legal_data_path, "rb") as f1:
        emb_legal_data = pickle.load(f1)
    return emb_legal_data

def load_bm25(bm25_path):
    with open(bm25_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    return bm25

def load_models(model_path):
    model = SentenceTransformer(model_path)
    return model

def load_question_json(question_path):
    question_data = json.load(open(question_path, "r"))
    return question_data
def load_model_bge(model_path):
    model = BGEM3FlagModel(model_path, use_fp16=True)
    return model
def encode_question_bge(question_data, model):
    print("Start encoding questions.")
    question_embs = []
    for _, item in tqdm(enumerate(question_data)):
        question = item["question"]
        emb_quest = model.encode(question, return_dense=True, return_sparse=True, return_colbert_vecs=True)
        question_embs.append(emb_quest)
    return question_embs

def encode_legal_data_bge(legal_dict_json, model):
    # Load phần document liên quan đến các điều luật
    doc_data = json.load(open(legal_dict_json))
    emb2_list = []
    for k, doc in tqdm(doc_data.items()):
        text_chunks = chunk_string(doc_data[k]["text"])
        emb_chunk = []
        for chunk in text_chunks:
            text_for_emb = doc_data[k]["title"] + " " + chunk
            emb_chunk.append(model.encode(text_for_emb, return_dense=True, return_sparse=True, return_colbert_vecs=True))
        emb2_list.append(emb_chunk)
    return emb2_list

def average_lists(data):
    keys = list(data.keys())
    count = len(keys)
    sum_lists = None
    for key in keys:
        current_lists = data[key]
        current_array = np.array(current_lists)
        if sum_lists is None:
            sum_lists = current_array
        else:
            sum_lists += current_array
    #avg_lists = sum_lists / count
    avg_lists = sum_lists
    return avg_lists


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data/data_test_new.json", type=str, help="for loading data test question")
    parser.add_argument("--raw_data", default="zac2021-ltr-data", type=str)
    parser.add_argument("--legal_dict_json", default="generated_data/legal_dict.json", type=str)
    parser.add_argument("--bm25_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--legal_data", default="saved_model/doc_refers_saved", type=str,
                        help="path to legal corpus for reference")
    parser.add_argument("--encode_legal_data", action="store_true", help="for legal data encoding")
    parser.add_argument("--using_reranking", type=int, help="Is using cross encoder to reranking")
    parser.add_argument("--top_k_acc", type=int, help="Get top K for evaluation")
    parser.add_argument("--using_bm25", type=int, help="Use bm25")
    parser.add_argument("--model_bge", type=str,default="/home/namnt/md1/NAMNT_DA2/checkpoint/checkpoint_bge_chunking/round2/checkpoint-5500",help="path to BGE model")
    args = parser.parse_args()

    top_n = 20
    # define path to model
    list_model_bi_encoder = [
        "/home/namnt/md1/NAMNT_DA2/checkpoint/ckp_sbert/round2/pretrain_mlm_chunking",
        "/home/namnt/md1/NAMNT_DA2/checkpoint/ckp_sbert/round2/pretrain_condenser_chunking",
        "/home/namnt/md1/sources/namnt/checkpoint/ckp_round2_cocondenser",
    ]
    print(f"Start loading model reranking")
    model_reranking = FlagReranker('/home/namnt/md1/NAMNT_DA2/checkpoint/checkpoint_reranking/checkpoint-5000_old', use_fp16=True)

    # load question from json file
    question_items = load_question_json(args.data)["items"]
    print("Number of questions: ", len(question_items))

    # load bm25 model
    bm25 = load_bm25(args.bm25_path)
    # load corpus to search
    print("Load legal data.")
    with open(args.legal_data, "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)

    predict_acc = []
    predict_recall = []

    predict_acc_l = []
    predict_recall_l = []

    # lưu score của từng mô hình trước khi ensemble
    dict_infor_score = {}
    list_lexical_score = []
    list_sparse_score = []
    # tính score của các mô hình bi-encoder
    for i,model_path in enumerate(list_model_bi_encoder):
        dict_infor_score[model_path] = []
        print("Start loading model bi-encoder.")
        model = load_models(model_path)
        # embedding questions
        question_embs = encode_question(question_items, model)

        #embedding passages
        emb_legal_data = encode_legal_data(args.legal_dict_json, model)

        for idx, item in tqdm(enumerate(question_items)):
            question_id = item["question_id"]
            question = item["question"]
            relevant_articles = item["relevant_info"]
            list_score = []
            emb_ques = question_embs[idx]
            for list_p_emb in emb_legal_data:
                list_score_p = []
                for p_emb in list_p_emb:
                    scores = util.cos_sim(emb_ques, p_emb)
                    list_score_p.append(scores.item())
                list_score.append(max(list_score_p))

            arr_score = np.array(list_score)
            arr_score = (arr_score - arr_score.min()) / (arr_score.max() - arr_score.min())
            if i == 0:
                dict_infor_score[model_path].append(0.15 * arr_score)
            if i == 1:
                dict_infor_score[model_path].append(0.1 * arr_score)
            if i == 2:
                dict_infor_score[model_path].append(0.15 * arr_score)

    dict_infor_score["dense"] = []
    dict_infor_score["colbert"] = []

    # Tính score của từng loai trong BGE-M3
    model_bge = load_model_bge(args.model_bge)
    question_embs = encode_question_bge(question_items, model_bge)
    emb_legal_data = encode_legal_data_bge(args.legal_dict_json, model_bge)
    for idx, item in tqdm(enumerate(question_items)):
        relevant_articles = item["relevant_info"]
        question_text = item["question"]
        list_score = []

        # compute score using BM25
        tokenized_query = bm25_tokenizer(question_text)
        doc_scores = bm25.get_scores(tokenized_query)

        # normalize score of BM25
        doc_scores_normalized = [item / max(doc_scores) for item in doc_scores]
        # embedding của question thu idx
        emb_ques = question_embs[idx]

        # loop qua tung emb trong emb_legal_data de tinh score
        list_score_dense = []
        list_score_sparse = []
        list_score_colbert = []
        for list_p_emb in emb_legal_data:
            list_score_p_dense = []
            list_score_p_sparse = []
            list_score_p_colbert = []
            for p_emb in list_p_emb:
                colbert_score = model_bge.colbert_score(emb_ques['colbert_vecs'], p_emb['colbert_vecs']).item()
                lexical_score = model_bge.compute_lexical_matching_score(emb_ques['lexical_weights'], p_emb['lexical_weights'])
                dense_score = emb_ques['dense_vecs'] @ p_emb['dense_vecs'].T

                list_score_p_dense.append(dense_score)
                list_score_p_sparse.append(lexical_score)
                list_score_p_colbert.append(colbert_score)
            list_score_dense.append(max(list_score_p_dense))
            list_score_sparse.append(max(list_score_p_sparse))
            list_score_colbert.append(max(list_score_p_colbert))

        # Normalize các score của BGE về khoảng (0,1)
        sum_arr = np.array([list_score_dense, list_score_sparse, list_score_colbert])
        arr_dense = np.array(list_score_dense)
        normalized_arr_dense = (arr_dense - sum_arr.min()) / (sum_arr.max() - sum_arr.min())
        arr_colbert = np.array(list_score_colbert)
        normalized_arr_colbert = (arr_colbert - sum_arr.min()) / (sum_arr.max() - sum_arr.min())
        arr_sparse = np.array(list_score_sparse)
        normalized_arr_sparse = (arr_sparse - sum_arr.min()) / (sum_arr.max() - sum_arr.min())

        # Tổng hợp lại score của dense và colbert
        dict_infor_score["dense"].append(0.3 * normalized_arr_dense)
        dict_infor_score["colbert"].append(0.3 * normalized_arr_colbert)

        #list_lexical_score.append(ensem_score_lexical)
        list_lexical_score.append(doc_scores_normalized)
        list_sparse_score.append(arr_sparse)


    # ensemble learning : trung bình lại các score
    avg_score = average_lists(dict_infor_score)

    # sort and compute score
    for idx, item in tqdm(enumerate(question_items)):
        relevant_articles = item["relevant_info"]
        question = item["question"]
        # Lấy ra score ensemble của các mô hình bi-encoder + colbert
        arr_score = avg_score[idx]

        # Lấy ra score của BM25
        doc_scores = list_lexical_score[idx]

        # using bm25
        if args.using_bm25 == 1:
            arr_score = doc_scores * arr_score
        else:
            arr_score = arr_score

        predictions = np.argpartition(arr_score, len(arr_score) - top_n)[-top_n:]
        new_scores = arr_score[predictions]

        sorted_ind = np.argsort(new_scores)[::-1]
        sorted_predictions = predictions[sorted_ind]
        sorted_new_scores = new_scores[sorted_ind]

        if args.using_reranking == 0:
            check_acc = 0
            check_recall = 0
            for i, idx_pred in enumerate(sorted_predictions):
                if i < args.top_k_acc:
                    pred = doc_refers[idx_pred]
                    for j, article in enumerate(relevant_articles):
                        if pred[0] == article["Field_id"] and pred[1] == article["infor_id"]:
                            check_acc += 1
                            check_recall += 1
                    if check_recall == len(relevant_articles):
                        break
                else:
                    break
            if check_acc >= 1:
                predict_acc.append(1)
            else:
                predict_acc.append(0)
            predict_recall.append(check_recall / len(relevant_articles))
        else:
            # Reranking compute score
            data_legal = json.load(open(args.legal_dict_json, "r"))
            data_legal = list(data_legal.items())
            legal_candidate = [data_legal[i] for i in sorted_predictions]
            legal_candidate_text = [item[1]["title"] + " " + item[1]["text"] for item in legal_candidate]
            reranking_score = get_reranking_score(question, legal_candidate_text, model_reranking)
            reranking_scores = [sigmoid(item) for item in reranking_score]


            final_score = []
            for i in range(len(reranking_scores)):
                final_score.append((math.sqrt((reranking_scores[i] ** 2 + sorted_new_scores[i]**2)/2)))

            sorted_indices = sorted(range(len(final_score)), key=lambda k: final_score[k], reverse=True)
            sorted_legal_candidate_text = [legal_candidate_text[i] for i in sorted_indices]
            sorted_final_score = [final_score[i] for i in sorted_indices]
            sorted_candidates = [sorted_predictions[i] for i in sorted_indices]

            check_2_acc = 0
            check_2_recall = 0
            for i,idx_pred in enumerate(sorted_candidates):
                if i < args.top_k_acc:
                    pred = doc_refers[idx_pred]
                    for article in relevant_articles:
                        if pred[0] == article["Field_id"] and pred[1] == article["infor_id"]:
                            check_2_acc += 1
                            check_2_recall += 1

                    if check_2_recall == len(relevant_articles):
                        break
                else:
                    break

            if check_2_acc >= 1:
                predict_acc.append(1)
            else:
                predict_acc.append(0)
            predict_recall.append(check_2_recall / len(relevant_articles))


    # In ra các câu hỏi mà dự đoán bị sai
    for i in range(len(predict_acc)):
        if predict_acc[i] < 1:
            print("Questin wrong:", question_items[i]["question"])

    acc_average = sum(predict_acc) / len(predict_acc)
    recall_average = sum(predict_recall) / len(predict_recall)
    print(f"Average accuracy: {acc_average}")
    print(f"Average recall: {recall_average}")

