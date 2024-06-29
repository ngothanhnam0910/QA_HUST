CUDA_VISIBLE_DEVICES=0 python train_sentence_bert.py --pretrained_model /home/gemai/md1/NAMNT_DA2/checkpoint/ckp_pretrain/ckp_mlm_ver2/checkpoint-4800 \
    --max_seq_length 256 \
    --pair_data_path /home/gemai/md1/NAMNT_DA2/data_train_sbert/pair_data/bm_25_pairs_top20 \
    --round 1 \
    --num_val 2000 \
    --epochs 7 \
    --saved_model /home/namnt/md1/NAMNT_DA2/checkpoint/ckp_sbert/type_loss/test_loss \
    --batch_size 16

# condenser
    CUDA_VISIBLE_DEVICES=1 python train_sentence_bert.py --pretrained_model /home/namnt/md1/NAMNT_DA2/checkpoint/ckp_pretrain/ckp_condenser/checkpoint-2800 \
    --max_seq_length 256 \
    --pair_data_path /home/gemai/md1/NAMNT_DA2/data_train_sbert/pair_data/bm_25_pairs_top20_chunking \
    --round 1 \
    --num_val 2000 \
    --epochs 15 \
    --saved_model /home/namnt/md1/NAMNT_DA2/checkpoint/ckp_sbert/round1/pretrain_condenser_chunking \
    --batch_size 16

# cocondenser
    CUDA_VISIBLE_DEVICES=1 python train_sentence_bert.py --pretrained_model /home/gemai/md1/NAMNT_DA2/checkpoint/ckp_pretrain/ckp_cocondenser_2/checkpoint-10000 \
    --max_seq_length 256 \
    --pair_data_path /home/gemai/md1/NAMNT_DA2/data_train_sbert/pair_data/bm_25_pairs_top20_chunking \
    --round 1 \
    --num_val 2000 \
    --epochs 10 \
    --saved_model /home/namnt/md1/NAMNT_DA2/checkpoint/ckp_sbert/round1/pretrain_cocondenser_chunking \
    --batch_size 16

CUDA_VISIBLE_DEVICES=1 python train_sentence_bert.py --pretrained_model /home/gemai/md1/sources/namnt/checkpoint/ckp_cocondenser/checkpoint-1400 \
    --max_seq_length 256 \
    --pair_data_path /home/gemai/md1/NAMNT_DA2/data_train_sbert/pair_data/bm_25_pairs_top20_chunking \
    --round 1 \
    --num_val 2000 \
    --epochs 10 \
    --saved_model /home/gemai/md1/sources/namnt/checkpoint/ckp_round1_cocondenser_ver2 \
    --batch_size 16