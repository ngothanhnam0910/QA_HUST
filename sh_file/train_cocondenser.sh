CUDA_VISIBLE_DEVICES=1 python Condenser/run_co_pre_training.py   \
    --output_dir /home/gemai/md1/NAMNT_DA2/checkpoint/ckp_pretrain/ckp_cocondenser/   \
    --model_name_or_path /home/gemai/md1/NAMNT_DA2/checkpoint/ckp_pretrain/ckp_condenser/checkpoint-2800   \
    --do_train   \
    --do_eval    \
    --save_steps 200   \
    --eval_steps 200  \
    --model_type bert   \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1   \
    --warmup_ratio 0.1   \
    --learning_rate 2e-5   \
    --num_train_epochs 50   \
    --dataloader_drop_last   \
    --overwrite_output_dir   \
    --dataloader_num_workers 4  \
    --n_head_layers 2   \
    --skip_from 6   \
    --max_seq_length 256   \
    --train_dir /home/gemai/md1/NAMNT_DA2/cocondenser_data/   \
    --weight_decay 0.01   \
    --late_mlm  \
    --cache_chunk_size 32 \
    --fp16
#    --remove_unused_columns false
#    --save_total_limit 1