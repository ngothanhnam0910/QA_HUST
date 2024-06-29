import json
import os
import re
from tqdm import tqdm
import argparse
import underthesea
def load_json(corpus_path):
    data = json.load(open(corpus_path))
    return data["items"]

def chunk_string(input_string):
    # Tách văn bản thành các câu
    sentences = underthesea.sent_tokenize(input_string)
    new_sentences = []
    for i in range(len(sentences)):
        if len(sentences[i]) > 2:
            if len(sentences[i - 1]) > 2:
                new_sentences.append(sentences[i])
            else:
                new_sentences.append(sentences[i - 1] + sentences[i])

    new_sentences = sentences
    chunks = []
    current_chunk = ""

    # Duyệt qua từng câu
    for sentence in sentences:
        # Nếu thêm câu vào chunk hiện tại không làm cho chunk vượt quá 200 từ
        if len(current_chunk.split()) + len(sentence.split()) <= 200:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data", type=str, help="path to training data")
    parser.add_argument("--save_dir", default="./generated_data", type=str, help="path to training data")
    args = parser.parse_args()
    os.makedirs(args.save_dir,exist_ok=True)
    cp = open(os.path.join(args.save_dir, "corpus.txt"), "w")
    corpus_path = os.path.join(args.data_dir, "template.json")

    data = json.load(open(corpus_path))

    save_dict = {}
    # co_f = open(os.path.join(args.save_dir, "cocondenser_data.json"), "w")
    count = 0
    for law_article in tqdm(data):
        law_id = law_article["Field_id"]
        law_articles = law_article["infor"]
        
        for sub_article in law_articles:
            article_id = sub_article["infor_id"]
            article_title = sub_article["title"]
            article_text = sub_article["text"]
            article_full = article_title + ". " + article_text
            article_full = article_full.replace("\n", " ")
            cp.write(article_full + "\n")
            
            # Save data for cocondenser 
            spans = [article_title]
            passages = re.split(r"\n[0-9]+\. |1\. ", article_text)
            for idx, p in enumerate(passages):
                if p != "":
                    article_full = article_title + ". " + p
                    article_full = article_full.replace("\n", " ")
                    spans.append(p)
            # co_f.write("#".join(spans) + "\n")
            concat_id = law_id.strip() + "_" + article_id.strip()
            if concat_id not in save_dict:
                count += 1

                text_chunks = chunk_string(article_text)
                save_dict[concat_id] = {"title": article_title, "text": text_chunks}
    
    # co_f.close()
    # print(count)
    # # exit()
    print("Create legal dict from raw data")
    with open(os.path.join(args.save_dir, "legal_dict_chunking.json"), "w", encoding='utf-8') as outfile:
        json.dump(save_dict, outfile ,ensure_ascii=False, indent=4)
    # print("Finish")
    # corpus_path_train = os.path.join(args.data_dir, "data_qa.json")
    # items = load_json(corpus_path_train)
    #
    # for item in tqdm(items):
    #     question = item["question"]
    #     cp.write(question + "\n")
    #
    # corpus_path_test = os.path.join(args.data_dir, "data_test.json")
    # items = load_json(corpus_path_test)
    #
    # for item in tqdm(items):
    #     question = item["question"]
    #     cp.write(question + "\n")
    #
    # cp.close()
