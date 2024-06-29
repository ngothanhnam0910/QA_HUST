 # fine tune 3 embedding
 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 -m FlagEmbedding.BGE_M3.run \
    --output_dir /home/gemai/md1/NAMNT_DA2/checkpoint/ckp_bge_round2 \
    --model_name_or_path /home/gemai/md1/NAMNT_DA2/checkpoint/ckp_bge/checkpoint-20000 \
    --train_data /home/gemai/md1/NAMNT_DA2/data_train_bge/bge_data_minedHN.json \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 2 \
    --dataloader_drop_last True \
    --normlized True \
    --temperature 0.02 \
    --query_max_len 128 \
    --passage_max_len 256 \
    --train_group_size 2 \
    --negatives_cross_device \
    --logging_steps 250 \
    --same_task_within_batch True \
    --unified_finetuning True \
    --use_self_distill True \
    --save_steps 500 \
    --query_instruction_for_retrieval ""

 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 -m FlagEmbedding.BGE_M3.run \
    --output_dir /home/namnt/md1/NAMNT_DA2/checkpoint/checkpoint_bge_chunking/round2 \
    --model_name_or_path /home/namnt/md1/NAMNT_DA2/checkpoint/checkpoint_bge_chunking/round1/checkpoint-10500 \
    --train_data /home/namnt/md1/NAMNT_DA2/data_train_bge/data_chunking_round2_ver2/ \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --dataloader_drop_last True \
    --normlized True \
    --temperature 0.02 \
    --query_max_len 128 \
    --passage_max_len 256 \
    --train_group_size 24 \
    --negatives_cross_device \
    --logging_steps 500 \
    --same_task_within_batch True \
    --unified_finetuning True \
    --use_self_distill True \
    --save_steps 500 \
    --query_instruction_for_retrieval "" \
    --gradient_checkpointing \
    --deepspeed '/home/namnt/DATN/zalo_ltr_2021/FlagEmbedding/examples/finetune/ds_config.json'
#    --save_total_limit 10