import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from rank_bm25 import *
from utils import bm25_tokenizer, load_json
import argparse

if __name__ == '__main__':

    file_name = "data_pretrain_bge"
    with open(os.path.join("/home/gemai/md1/NAMNT_DA2/generated_data", file_name + '.json'), 'w') as output_file:
        with open("/home/gemai/md1/NAMNT_DA2/generated_data/corpus.txt", 'r') as input_file:
            lines = input_file.readlines()

            for line in tqdm(lines):
                # create_data for bge
                save_dict = {}
                save_dict["text"] = line
                # lưu save_dict vao file output
                output_file.write(json.dumps(save_dict,ensure_ascii=False) + '\n')
