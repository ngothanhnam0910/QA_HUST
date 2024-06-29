import os
import google.generativeai as genai
import pickle
import json
import csv
# from IPython.display import display
# from IPython.display import Markdown
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap

#os.environ["GOOGLE_API_KEY"] = "AIzaSyAfNSnqj948ySOhnF67HZaY2GPz8XNZb_A"
#os.environ["GOOGLE_API_KEY"] = "AIzaSyCY1-ZqrwO6MQMI2p2fGM8KQotbNjIFGYQ"
os.environ["GOOGLE_API_KEY"] = "AIzaSyCFf1q3msamNu-9ix8tn9uNIcd1cNPL-Ig"

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# Xử lý relevant documents
def get_context(relevant_documents):
  context = "\n\n".join(relevant_documents)
  return context
def generate_answer(question, list_relevant):
    # Define template
    template = """Trả lời câu hỏi bằng một câu đầy đủ, dựa trên ngữ cảnh sau đây:
    {context}
    Câu hỏi: {question}
    """
    # template = """ Là một chatbot thông minh, hãy giúp tôi trả lời câu hỏi dựa trên ngữ cảnh được cung cấp
    # Dưới đây là một ví dụ trước khi bạn thực hiện :
    # Câu hỏi: Bạn A đang học ở đâu?
    # Ngữ cảnh: Bạn A đang học năm 3 ở Đại học Bách Khoa Hà Nội.
    # Câu trả lời: Bạn A học ở Đại học Bách Khoa Hà Nội.
    #
    # Dựa vào hướng dẫn ở trên, hãy giúp tôi trả lời câu hỏi dựa vào ngữ cảnh:
    # Ngữ cảnh: {context}
    # Câu hỏi: {question}
    # """
    # template = """
    # Dựa vào hướng dẫn ở trên, hãy giúp tôi trả lời câu hỏi dựa vào ngữ cảnh:
    # Hãy suy nghĩ từng bước một: ví dụ có những câu bạn sẽ cần trích ra các giá trị số đem đi so sánh với câu hỏi để suy ra câu trả lời (VD: 4>2)
    # Ngữ cảnh: {context}
    # Câu hỏi: {question}
    # """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.1)
    output_parser = StrOutputParser()
    chain = RunnableMap({
        "context": lambda x: get_context(x["relevant_documents"]),  # Sử dụng hàm get_context để tạo ngữ cảnh
        "question": lambda x: x["question"]
    }) | prompt | model | output_parser
    answer = chain.invoke(
        {"question": question,
         "relevant_documents": list_relevant,
        }
    )
    return answer

if __name__ == "__main__":

    #Load test dataset
    print("load test dataset")
    load_data_test = json.load(open("data/data_test_new.json",'r'))
    list_data_test = load_data_test["items"]

    # Load template
    print("load template")
    load_template = json.load(open("data/template.json",'r'))

    # luu list question + list answer
    list_question = []
    list_answer = []

    for test_item in list_data_test[0:35]:
        field_id = test_item["relevant_info"][0]["Field_id"]
        infor_id = test_item["relevant_info"][0]["infor_id"]
        question = test_item["question"]
        list_question.append(question)

        for item_template in load_template:
            if item_template["Field_id"] == field_id:
                for infor in item_template["infor"]:
                    if infor["infor_id"] == infor_id:
                        text = infor["text"]
                        # print(f"question: {question}")
                        # print(f"text: {text}")
                        # exit()
                        answer = generate_answer(question, [text])
                        list_answer.append(answer)

                    else:
                        continue
                break

            else:
                continue

    with open('test_output_answer/output.csv', mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Answer'])  # Ghi header cho các cột
        for item1, item2 in zip(list_question, list_answer):
            writer.writerow([item1, item2])
    print("finished generating")
