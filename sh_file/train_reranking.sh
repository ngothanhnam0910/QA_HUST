#CUDA_VISIBLE_DEVICES=0 python train_reranking.py \
#      --pretrained_model /home/gemai/md1/NAMNT_DA2/checkpoint/ckp_sbert/round2/pretrain_cocondenser \
#      --pair_data_path /home/gemai/md1/NAMNT_DA2/data_train_sbert/pair_data/save_pairs_mlm_top20.pkl \
#      --epochs 10 \
#      --saved_model /home/gemai/md1/NAMNT_DA2/checkpoint/reranking_model \
#      --batch_size  16

 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 -m FlagEmbedding.reranker.run \
 --output_dir /home/namnt/md1/NAMNT_DA2/checkpoint/checkpoint_reranking/ckp_chunking_2 \
 --model_name_or_path BAAI/bge-reranker-v2-m3 \
 --train_data /home/namnt/md1/NAMNT_DA2/data_reranking_bge/data_16_05.json \
 --learning_rate 1e-5 \
 --fp16 \
 --num_train_epochs 10 \
 --per_device_train_batch_size 1 \
 --gradient_accumulation_steps 1 \
 --dataloader_drop_last True \
 --train_group_size 24 \
 --max_len 256  \
 --weight_decay 0.01 \
 --logging_steps 250