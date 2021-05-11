import pickle
import torch
from transformers import RobertaTokenizer, RobertaModel

# 加载模型
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("python_model")

# 查询语句向量
query = "reverse an array"
query_vec = model(tokenizer(query, return_tensors='pt')['input_ids'])[1]

# 取回code vec
with open('code_vecs.pkl','rb') as vec_input_files:
    new_code_vectors = pickle.load(vec_input_files)

# 取回data
with open('data.pkl','rb') as data_input_files:
    new_search_data = pickle.load(data_input_files)

# 评分
scores = torch.einsum("ab,cb->ac", query_vec, torch.tensor(new_code_vectors).float())
scores = torch.softmax(scores, -1)

# 取前k个结果
scores_clone = scores.clone() # 对列表进行浅复制，避免后面更改原列表数据
List = scores_clone.detach().numpy().flatten().tolist()
k = 5
index_k = []
for i in range(k):
    index_i = List.index(max(List))  # 得到列表的最大值，并得到该最大值的索引
    index_k.append(index_i)  # 记录最小值索引
    List[index_i] = -1  # 将遍历过的列表最大值改为-1，下次不再选择

# 输出
for i in range(k):
    index = index_k[i]
    print("Code:", new_search_data[index]['code'])
    print("Score:", scores[0, index].item(),'/n')


