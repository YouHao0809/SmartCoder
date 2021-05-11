import pickle
import time

import torch
from flask import Flask, jsonify, request
from transformers import RobertaModel, RobertaTokenizer

# 加载模型
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("python_model")
# 取回 code_vectors
with open('code_vecs.pkl', 'rb') as vec_input_files:
    new_code_vectors = pickle.load(vec_input_files)
# 取回 data
with open('data.pkl', 'rb') as data_input_files:
    new_search_data = pickle.load(data_input_files)


def create_app():
    @app.route("/")
    def index():
        return jsonify(hello="world")

    @app.route("/search_code", methods=["POST"])
    def search():
        if request.method == "POST":
            payload = request.get_json()
            query = payload["query"]

            t0 = time.time()
            query_vec = model(tokenizer(query, return_tensors='pt')['input_ids'])[1]
            t1 = time.time()

            # 评分
            scores = torch.einsum("ab,cb->ac", query_vec, torch.tensor(new_code_vectors).float())
            scores = torch.softmax(scores, -1)

            # 取前k个结果
            scores_clone = scores.clone()  # 对列表进行浅复制，避免后面更改原列表数据
            List = scores_clone.detach().numpy().flatten().tolist()
            k = 5
            index_k = []
            for i in range(k):
                index_i = List.index(max(List))  # 得到列表的最大值，并得到该最大值的索引
                index_k.append(index_i)  # 记录最小值索引
                List[index_i] = -1  # 将遍历过的列表最大值改为-1，下次不再选择

            # 输出
            output = []
            for i in range(k):
                index = index_k[i]
                output.append([new_search_data[index]['code'], scores[0, index].item()])
                # print("Code:", new_search_data[index]['code'])
                # print("Score:", scores[0, index].item(), '/n')

            device = "cuda" if torch.cuda.is_available() else "cpu"
            result = {
                'output': output,
                'time': (t1 - t0),
                'device': device,
            }
            return jsonify(**result)

    @app.route("/tokenizer", methods=["POST"])
    def get_vector():
        if request.method == "POST":
            payload = request.get_json()
            tokens = model(tokenizer(payload["query"], return_tensors='pt')['input_ids'])[1].detach().numpy().tolist()
            return jsonify(tokens=tokens)

    return app


app = Flask(__name__)
app = create_app()
app.run(host='127.0.0.1', port=20006)

# 退出时安全kill
