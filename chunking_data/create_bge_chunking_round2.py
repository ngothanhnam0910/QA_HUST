import json
from FlagEmbedding import FlagReranker
import math
import os
import argparse
def sigmoid(x):
    return 1/(1 + math.exp(-x))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/namnt/md1/NAMNT_DA2/data_train_bge/data_chunking_round2/toy_finetune_data_minedHN.json", type=str, help="path to input data")
    parser.add_argument("--save_pair_path", default="/home/namnt/md1/NAMNT_DA2/data_train_bge/data_chunking_round2_ver2", type=str,help="path to save pair sentence directory")
    args = parser.parse_args()

    print(f"Starting load rerannker model")
    reranker = FlagReranker('/home/namnt/md1/NAMNT_DA2/checkpoint/checkpoint_reranking/ckp_chunking_2/checkpoint-9500', use_fp16=True)
    print(f"checkpoint-9500")
    if not os.path.exists(args.save_pair_path):
        os.mkdir(args.save_pair_path)

    # Start load data
    file_name = "data_16_05"
    with open(os.path.join(args.save_pair_path, file_name + '.json'), 'w') as output_file:
        with open(args.data_path, 'r') as f:
            for line in f:
                # Loại bỏ ký tự xuống dòng (`\n`) ở cuối mỗi dòng (nếu cần)
                read_line = json.loads(line.rstrip())
                read_line.pop("pos_scores")
                read_line.pop("neg_scores")
                read_line["pos_scores"] = []
                read_line["neg_scores"] = []
                query = read_line["query"]
                for pos_sentence in read_line["pos"]:
                    scores = reranker.compute_score([pos_sentence, query])
                    read_line["pos_scores"].append((scores))
                for neg_sentence in read_line["neg"]:
                    scores = reranker.compute_score([neg_sentence, query])
                    read_line["neg_scores"].append((scores))
                # print(f"positive score: ",read_line["pos_scores"])
                # print(f"negative score: ", read_line["neg_scores"])
                # exit()
                output_file.write(json.dumps(read_line,ensure_ascii=False) + '\n')

    print(f"write sucessful")


