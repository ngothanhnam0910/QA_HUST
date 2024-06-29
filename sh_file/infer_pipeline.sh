CUDA_VISIBLE_DEVICES=0 python new_predict.py  \
--legal_dict_json /home/gemai/md1/NAMNT_DA2/generated_data/legal_dict.json \
--bm25_path /home/gemai/md1/NAMNT_DA2/save_bm25/bm25_Plus_04_06_model_full_manual_stopword  \
--legal_data /home/gemai/md1/NAMNT_DA2/save_bm25/doc_refers_saved \
--encode_legal_data \
--top_k_acc 3 \
--using_reranking 0 \
--using_bm25 0