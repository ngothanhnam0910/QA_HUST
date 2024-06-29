import numpy as np
import json
import torch
from tqdm import tqdm
from rank_bm25 import *
import argparse
import os
import pickle
from utils_bm25 import bm25_tokenizer
import time
from sentence_transformers import SentenceTransformer, util
from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding import FlagReranker
import math
import time

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def encode_legal_data(legal_dict_json, model):
    # Load phần document liên quan đến các điều luật
    doc_data = json.load(open(legal_dict_json))
    emb2_list = []
    for k, doc in tqdm(doc_data.items()):
        text_for_emb = doc_data[k]["title"] + " " + doc_data[k]["text"]
        emb2_list.append(model.encode(text_for_emb, return_dense=True, return_sparse=True, return_colbert_vecs=True))
    return emb2_list

def encode_question(question_data, model):
    print("Start encoding questions.")
    question_embs = []

    for _, item in tqdm(enumerate(question_data)):
        question = item["question"]
        emb_quest = model.encode(question, return_dense=True, return_sparse=True, return_colbert_vecs=True)
        question_embs.append(emb_quest)
    return question_embs


def load_encoded_legal_corpus(legal_data_path):
    print("Start loading legal corpus.")
    with open(legal_data_path, "rb") as f1:
        emb_legal_data = pickle.load(f1)
    return emb_legal_data


def load_model(model_path):
    model = BGEM3FlagModel(model_path, use_fp16=True)
    return model

def load_bm25(bm25_path):
    with open(bm25_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    return bm25

def load_question_json(question_path):
    question_data = json.load(open(question_path, "r"))
    return question_data

def get_reranking_score(question,text_legal_data, model):
    list_reranking_score = model.compute_score([[question,item] for item in text_legal_data])
    return list_reranking_score

def get_relavance_passage(question, topk):

    top_n = 10
    # Load model bge
    path_model_bge = "/home/namnt/md1/NAMNT_DA2/checkpoint/checkpoint_fine_tune2/checkpoint-1000"
    model_encode = load_model(path_model_bge)

    # Load model reranking
    print(f"Start loading model reranking")
    model_reranking = FlagReranker('/home/namnt/md1/NAMNT_DA2/checkpoint/checkpoint_reranking/checkpoint-5000_old', use_fp16=True)

    # load bm25 model
    bm25 = load_bm25("/home/namnt/md1/NAMNT_DA2/save_bm25/bm25_Plus_04_06_model_full_manual_stopword")

    # compute score using BM25
    #time1 = time.time()
    tokenized_query = bm25_tokenizer(question)
    doc_scores = bm25.get_scores(tokenized_query)
    #time2 = time.time()
    #print(time2 - time1)
    #exit()


    # load corpus to search
    print("Load legal data.")
    with open("/home/namnt/md1/NAMNT_DA2/save_bm25/doc_refers_saved", "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)

    print("Load embedding legal data")

    # load pre encoded for legal corpus
    # if args.encode_legal_data:
    #emb_legal_data = encode_legal_data("/home/namnt/md1/NAMNT_DA2/generated_data/legal_dict.json", model_encode)
    emb_legal_data = load_encoded_legal_corpus("/home/namnt/md1/NAMNT_DA2/embedding_data/bge/bge_emb.pkl")

    # encode question
    question_embs = model_encode.encode(question, return_dense=True, return_sparse=True, return_colbert_vecs=True)

    list_score = []
    # loop qua tung emb trong emb_legal_data de tinh score
    for p_emb in emb_legal_data:
        colbert_score = model_encode.colbert_score(question_embs['colbert_vecs'], p_emb['colbert_vecs']).item()
        lexical_score = model_encode.compute_lexical_matching_score(question_embs['lexical_weights'], p_emb['lexical_weights'])
        dense_score = question_embs['dense_vecs'] @ p_emb['dense_vecs'].T

        final_score = (lexical_score * 0.4 + dense_score * 0.6)
        list_score.append(final_score)

    arr_score = np.array(list_score)
    doc_scores = [item/max(doc_scores) for item in doc_scores]
    arr_score = doc_scores * arr_score

    predictions = np.argpartition(arr_score, len(arr_score) - top_n)[-top_n:]
    new_scores = arr_score[predictions]

    sorted_ind = np.argsort(new_scores)[::-1]
    sorted_predictions = predictions[sorted_ind]
    sorted_new_scores = new_scores[sorted_ind]


    data_legal = json.load(open("/home/namnt/md1/NAMNT_DA2/generated_data/legal_dict.json", "r"))
    data_legal = list(data_legal.items())
    legal_candidate = [data_legal[i] for i in sorted_predictions]
    legal_candidate_text = [item[1]["title"] + " " + item[1]["text"] for item in legal_candidate]

    #time1 = time.time()
    reranking_score = get_reranking_score(question, legal_candidate_text, model_reranking)
    time2 = time.time()
    #print(time2 - time1)
    #exit()
    reranking_scores = [sigmoid(item) for item in reranking_score]


    final_score = []
    for i in range(len(reranking_scores)):
        final_score.append((math.sqrt((reranking_scores[i] ** 2 + sorted_new_scores[i] ** 2) / 2)))

    sorted_indices = sorted(range(len(final_score)), key=lambda k: final_score[k], reverse=True)
    sorted_legal_candidate_text = [legal_candidate_text[i] for i in sorted_indices]
    sorted_final_score = [final_score[i] for i in sorted_indices]
    sorted_candidates = [sorted_predictions[i] for i in sorted_indices]

    list_relevance = []
    for i,candidate in enumerate(sorted_candidates):
        if i < topk:
            list_relevance.append(doc_refers[candidate][2])
    # print(list_relevance)
    return list_relevance


if __name__ == "__main__":
    question = "Trường Đại học Bách khoa thành lập năm bao nhiu?"
    topk = 3
    get_relavance_passage(question, topk)
