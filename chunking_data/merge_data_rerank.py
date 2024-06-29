import json
import os

mlm_reranking_path = "/home/namnt/md1/NAMNT_DA2/data_reranking_bge/data_mlm_reranking.json"
condenser_reranking_path = "/home/namnt/md1/NAMNT_DA2/data_reranking_bge/data_condeser_reranking.json"
cocondenser_reranking_path = "/home/namnt/md1/NAMNT_DA2/data_reranking_bge/data_cocondeser_reranking.json"

# Start load data
save_pair_path = "/home/namnt/md1/NAMNT_DA2/data_reranking_bge"
file_name = "final_reranking"
with open(os.path.join(save_pair_path, file_name + '.json'), 'w') as output_file:
    f_mlm = open(mlm_reranking_path,"r")
    f_condenser = open(condenser_reranking_path,"r")
    f_cocondenser = open(cocondenser_reranking_path, "r")
    for line_1, line_2 ,line_3 in zip(f_mlm,f_condenser,f_cocondenser):
        save_dict = {}
        read_line_1 = json.loads(line_1.rstrip())
        read_line_2 = json.loads(line_2.rstrip())
        read_line_3 = json.loads(line_3.rstrip())
        save_dict["query"] = read_line_1["query"]
        save_dict["pos"] = read_line_1["pos"]
        save_dict["neg"] = list(set(read_line_1["neg"] + read_line_2["neg"] + read_line_3["neg"]))

        output_file.write(json.dumps(save_dict, ensure_ascii=False) + '\n')
        print(f"write sucessful")