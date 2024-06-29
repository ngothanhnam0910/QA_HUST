CUDA_VISIBLE_DEVICES=1 python predict_bge.py \
 --legal_dict_json /home/gemai/md1/NAMNT_DA2/generated_data/legal_dict.json \
 --legal_data /home/gemai/md1/NAMNT_DA2/save_bm25/doc_refers_saved \
 --encode_legal_data \
 --topk 3 \
 --model_path /home/gemai/md1/NAMNT_DA2/checkpoint/ckp_bge/checkpoint-20000


 CUDA_VISIBLE_DEVICES=1 python predict_bge.py \
 --legal_dict_json /home/gemai/md1/NAMNT_DA2/generated_data/legal_dict.json \
 --legal_data /home/gemai/md1/NAMNT_DA2/save_bm25/doc_refers_saved \
 --encode_legal_data \
 --topk 3 \
 --model_path /home/gemai/md1/NAMNT_DA2/checkpoint/ckp_bge_distillation/checkpoint-4000