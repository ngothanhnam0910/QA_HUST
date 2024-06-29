export MODEL_NAME=vinai/phobert-large
export MAX_LENGTH=256
export DATA_FILE=/home/gemai/md1/NAMNT_DA2/generated_data/corpus.txt
export SAVE_CONDENSER=/home/gemai/md1/NAMNT_DA2/condenser_data/

python Condenser/helper/create_train.py --tokenizer_name $MODEL_NAME --file $DATA_FILE --save_to $SAVE_CONDENSER

