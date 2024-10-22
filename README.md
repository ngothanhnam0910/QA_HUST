# DATN - Hệ thống hỏi đáp đại học Bách Khoa Hà Nội
Source code

# Data
- Dữ liệu được thu thập trên các trang thông tin chính thức của ĐHBKHN => thu được từng văn bản điều luật
- Sử dụng GPT4 để sinh ra các câu hỏi cho từng văn bản điều luật => bộ dữ liệu truy xuất
- Dữ liệu sau khi làm sạch gồm 2 file:
    - template.json: chứa các văn bản điều luật
    - data_qa.json: tập dữ liệu truy xuất

# Tổng quan hệ thống:
Luồng hoạt động sẽ như trong hình vẽ
<p align="center">
    <img src="figs/gen_ansIr.jpg">
</p> 

# Luồng hoạt động của mô-đun truy xuất
<p align="center">
    <img src="figs/Retrieval.jpg">
</p>

# Huấn luyện retriever
<p align="center">
    <img src="figs/Retriever.jpg">
</p>

=

## Create Folder

Create a new folder for generated data for training ``mkdir generated_data``

## Train BM 25
To train BM25: ``python bm25_train.py``
Use load_docs to save time for later run: ``python bm25_train.py --load_docs``

To evaluate: ``python bm25_create_pairs.py``
This step will also create top_k negative pairs from BM25. We choose top_k= 20
Pairs will be saved to: pair_data/

These pairs will be used to train round 1 Sentence Transformer model

## Create corpus: 
Run ``python create_corpus.txt``
This step will create:
- corpus.txt  (for finetune language model)
- cocondenser_data.json (for finetune CoCondenser model)

## Finetune language model using Huggingface
Pretrained model:
- phobert-large: vinai/phobert-large

$MODEL_NAME= phobert-large
$DATA_FILE= corpus.txt
$SAVE_DIR= /path/to/your/save/directory

Run the following cmd to train Masked Language Model:
```
python run_mlm.py \
    --model_name_or_path $MODEL_NAME \
    --train_file $DATA_FILE \
    --do_train \
    --do_eval \
    --output_dir $SAVE_DIR \
    --line_by_line \
    --overwrite_output_dir \
    --save_steps 2000 \
    --num_train_epochs 20 \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 32
```

##  Train condenser and cocondenser from language model checkpoint
Original source code here: https://github.com/luyug/Condenser (we modified several lines of code to make it compatible with current version of transformers)

### Create data for Condenser: 
 
```
python helper/create_train.py --tokenizer_name $MODEL_NAME --file $DATA_FILE --save_to $SAVE_CONDENSER \ --max_len $MAX_LENGTH 

$MODEL_NAME=vinai/phobert-large
$MAX_LENGTH=256
$DATA_FILE=../generated_data/corpus.txt
$SAVE_CONDENSER=../generated_data/
```

$MODEL_NAME checkpoint from finetuned language model 

```
python run_pre_training.py \
  --output_dir $OUTDIR \
  --model_name_or_path $MODEL_NAME \
  --do_train \
  --save_steps 2000 \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $ACCUMULATION_STEPS \
  --fp16 \
  --warmup_ratio 0.1 \
  --learning_rate 5e-5 \
  --num_train_epochs 8 \
  --overwrite_output_dir \
  --dataloader_num_workers 32 \
  --n_head_layers 2 \
  --skip_from 6 \
  --max_seq_length $MAX_LENGTH \
  --train_dir $SAVE_CONDENSER \
  --weight_decay 0.01 \
  --late_mlm
```
We use this setting to run Condenser:
```
python run_pre_training.py   \
    --output_dir saved_model_1/  \
    --model_name_or_path ../Legal_Text_Retrieval/lm/large/checkpoint-30000   \
    --do_train   
    --save_steps 2000   \
    --per_device_train_batch_size 32   \
    --gradient_accumulation_steps 4   \
    --fp16   \
    --warmup_ratio 0.1   \
    --learning_rate 5e-5   \
    --num_train_epochs 8   \
    --overwrite_output_dir   \
    --dataloader_num_workers 32   \
    --n_head_layers 2   \
    --skip_from 6   \
    --max_seq_length 256   \
    --train_dir ../generated_data/   \
    --weight_decay 0.01   \
    --late_mlm
```


## Train cocodenser:
First, we create data for cocodenser

```
python helper/create_train_co.py \
    --tokenizer vinai/phobert-large \
    --file ../generated_data/cocondenser/corpus.txt.json \
    --save_to data/large_co/corpus.txt.json \
```

Run the following cmd to train co-condenser model:
```
python  run_co_pre_training.py   \
    --output_dir saved_model/cocondenser/   \
    --model_name_or_path $CODENSER_CKPT   \
    --do_train   \
    --save_steps 2000   \
    --model_type bert   \
    --per_device_train_batch_size 32   \
    --gradient_accumulation_steps 1   \
    --fp16   \
    --warmup_ratio 0.1   \
    --learning_rate 5e-5   \
    --num_train_epochs 10   \
    --dataloader_drop_last   \
    --overwrite_output_dir   \
    --dataloader_num_workers 32   \
    --n_head_layers 2   \
    --skip_from 6   \
    --max_seq_length 256   \
    --train_dir ../generated_data/cocondenser/   \
    --weight_decay 0.01   \
    --late_mlm  \
    --cache_chunk_size 32 \
    --save_total_limit 1
```

## Train bi-encoder
### Round 1: using negative pairs of sentence generated from BM25
For each Masked Language Model, trained a sentence transformer corresponding to it
Run the following command to train round 1 of sentence bert model

Note: Use cls_pooling for condenser and cocodenser
```
python train_sentence_bert.py 
    --pretrained_model /path/to/your/pretrained/mlm/model\
    --max_seq_length 256 \
    --pair_data_path /path/to/your/negative/pairs/data\
    --round 1 \
    --num_val $NUM_VAL\
    --epochs 10\
    --saved_model /path/to/your/save/model/directory\
    --batch_size 32\
```


### Round 2: using hard negative pairs create from Round 1 model
- Step 1: Run the following cmd to generate hard negative pairs from round 1 model:
```
python hard_negative_mining.py \
    --model_path /path/to/your/sentence/bert/model\
    --data_path /path/to/the/lagal/corpus/json\
    --save_path /path/to/directory/to/save/neg/pairs\
    --top_k top_k_negative_pair
```
Pick top k is 20 .
- Use the data generated from step 1 to train round 2 of sentence bert model for each model from round 1:
To train round 2, please use the following command:
```
python train_sentence_bert.py 
    --pretrained_model /path/to/your/pretrained/mlm/model\
    --max_seq_length 256 \
    --pair_data_path /path/to/your/negative/pairs/data\
    --round 2 \
    --num_val $NUM_VAL\
    --epochs 5\
    --saved_model /path/to/your/save/model/directory\
    --batch_size 32\
```
Tips: Use small learning rate for model convergence

## Finetune BGE
```
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
```

## Run Streamlit
```
streamlit run generate_question/test_streamlit.py
```

