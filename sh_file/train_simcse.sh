CUDA_VISIBLE_DEVICES=0 python SimCSE/train_simCSE.py \
      --model_name BAAI/bge-m3-unsupervised \
      --data_path /home/gemai/md1/NAMNT_DA2/generated_data/corpus.txt \
      --saved_model /home/gemai/md1/NAMNT_DA2/checkpoint/simcse

