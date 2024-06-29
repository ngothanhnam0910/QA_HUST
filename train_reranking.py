from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import os
import argparse
import pickle
from torch import nn
def load_pair_data(pair_data_path):
    with open(pair_data_path, "rb") as pair_file:
        pairs = pickle.load(pair_file)
    return pairs



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", default="", type=str, help="path to your language model")
    parser.add_argument("--pair_data_path", type=str, default="/save_pairs_finetuned_phobert_top20.pkl", help="path to saved pair data")
    parser.add_argument("--num_val", default=1000, type=int, help="number of eval data")
    parser.add_argument("--epochs", default=10, type=int, help="Number of training epochs")
    parser.add_argument("--saved_model", default="/ckp_finetuned_sentencebert_round2", type=str, help="path to savd model directory.")
    parser.add_argument("--batch_size", type=int, default= 4, help="batch size")
    args = parser.parse_args()
    #### Just some code to print debug information to stdout
    logging.basicConfig(
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
    )
    logger = logging.getLogger(__name__)

    logger.info("Read  train dataset")

    save_pairs = load_pair_data(args.pair_data_path)
    # save_pairs_1 = load_pair_data("/home/gemai/md1/NAMNT_DA2/data_train_sbert/pair_data/save_pairs_mlm_top20.pkl")
    # save_pairs_2 = load_pair_data("/home/gemai/md1/NAMNT_DA2/data_train_sbert/pair_data/save_pairs_condenser_top20.pkl")
    # save_pairs_3 = load_pair_data("/home/gemai/md1/NAMNT_DA2/data_train_sbert/pair_data/save_pairs_cocondenser_top20.pkl")
    # combined_list = save_pairs_1 + save_pairs_2 + save_pairs_3

    # Loại bỏ các phần tử giống nhau hoàn toàn
    # save_pairs = []
    # for item in combined_list:
    #     if item not in save_pairs:
    #         save_pairs.append(item)
    print(f"There are {len(save_pairs)} pair sentences.")
    train_samples = []
    dev_samples = []

    num_train = len(save_pairs) - args.num_val

    for idx, pair in enumerate(save_pairs):
        relevant = float(pair["relevant"])
        question = pair["question"]
        document = pair["document"]
        example = InputExample(texts=[question, document], label=relevant)
        if idx <= num_train:
            train_samples.append(example)
        else:
            dev_samples.append(example)


    # Define our CrossEncoder model. Using model
    model = CrossEncoder(args.pretrained_model, num_labels=1,  max_length= 256)

    # We wrap train_samples, which is a list of InputExample, in a pytorch DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)

    # During training, we use BinaryClassificationEvaluator to measure the performance on the dev set
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name="AllNLI-dev")

    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * 0.1)  # 10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))


    # Train the model
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs= args.epochs,
        optimizer_params={'lr': 1e-6},
        evaluation_steps=500,
        warmup_steps= 5000,
        output_path=args.saved_model,
        save_best_model= True,
        use_amp=True,
        show_progress_bar= True
    )