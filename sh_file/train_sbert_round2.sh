CUDA_VISIBLE_DEVICES=1 python train_sentence_bert.py --pretrained_model /home/gemai/md1/NAMNT_DA2/checkpoint/ckp_sbert/round1/pretrain_mlm_chunking \
    --max_seq_length 256 \
    --pair_data_path /home/gemai/md1/NAMNT_DA2/data_train_sbert/pair_data/save_pairs_mlm_top20_chunking.pkl \
    --round 2 \
    --num_val 2000 \
    --epochs 10 \
    --saved_model /home/gemai/md1/NAMNT_DA2/checkpoint/ckp_sbert/round2/pretrain_mlm_chunking \
    --batch_size 16

# condenser
CUDA_VISIBLE_DEVICES=1 python train_sentence_bert.py --pretrained_model /home/gemai/md1/NAMNT_DA2/checkpoint/ckp_sbert/round1/pretrain_condenser_chunking \
    --max_seq_length 256 \
    --pair_data_path /home/gemai/md1/NAMNT_DA2/data_train_sbert/pair_data/save_pairs_condenser_top20_chunking.pkl \
    --round 2 \
    --num_val 2000 \
    --epochs 8 \
    --saved_model /home/gemai/md1/NAMNT_DA2/checkpoint/ckp_sbert/round2/pretrain_condenser_chunking \
    --batch_size 16

# condenser
CUDA_VISIBLE_DEVICES=1 python train_sentence_bert.py --pretrained_model /home/namnt/md1/sources/namnt/checkpoint/ckp_round1_cocondenser_ver2 \
    --max_seq_length 256 \
    --pair_data_path /home/gemai/md1/NAMNT_DA2/data_train_sbert/pair_data/save_pairs_cocondenser_top20_chunking_ver2.pkl \
    --round 2 \
    --num_val 2000 \
    --epochs 10 \
    --saved_model /home/namnt/md1/sources/namnt/checkpoint/ckp_round2_cocondenser \
    --batch_size 16