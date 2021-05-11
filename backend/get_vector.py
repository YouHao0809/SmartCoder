import json
import os
import pickle
import time

import numpy as np
from transformers import RobertaTokenizer, RobertaModel

# 加载模型
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("python_model")
# 查询语句向量
query = "set a variable as hello world"
query_vec = model(tokenizer(query, return_tensors='pt')['input_ids'])[1]
# 加载文件到对应数据集合
for language in ['python']:
    print(language)
    train, valid, test, codebase = [], [], [], []
    for root, dirs, files in os.walk(language):
        for file in files:
            temp = os.path.join(root, file)
            if '.jsonl' in temp:
                if 'train' in temp:
                    pass
                    # train.append(temp)
                elif 'valid' in temp:
                    valid.append(temp)
                    codebase.append(temp)
                elif 'test' in temp:
                    test.append(temp)
                    codebase.append(temp)
    # 读取文件获取数据字典
    train_data, valid_data, test_data, codebase_data = {}, {}, {}, {}
    for files, data in [[train, train_data], [valid, valid_data], [test, test_data], [codebase, codebase_data]]:
        for file in files:
            with open(file) as f:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    data[js['url']] = js

    print("训练数据", len(train_data))
    print("验证数据", len(valid_data))
    print("测试数据", len(test_data))
    print("base数据", len(codebase_data))
    # print(type(train_data))
    # print(train_data['https://github.com/smdabdoub/phylotoast/blob/0b74ef171e6a84761710548501dfac71285a58a3/phylotoast/util.py#L311-L334']['language'])

    # 选出部分数据
    search_data = {}
    clock = 0
    for key in test_data.keys():
        if clock > 7:
            break
        search_data[key] = test_data[key]
        clock += 1
    final_data = list(search_data.values())
    # code转为向量转为（len(data),768）ndarray
    code_vectors = np.empty((1, 768))

    # 单行转换
    # for value in final_data:
    #     # 开始时间
    #     time_start = time.time()
    #     # code向量=>ndarray
    #     code_vec = model(tokenizer(value['code'], return_tensors='pt', truncation=True)['input_ids'])[1]
    #     code_vectors = np.append(code_vectors, code_vec.detach().numpy(), axis = 0)
    #     # 计时器
    #     time_end = time.time()
    #     time_cost = time_end - time_start
    #     print('time cost', time_cost, 's')

    # 按batch转换
    clock = 0
    values = []
    batch_size = 4
    for value in final_data:
        clock += 1
        values.append(value['code'])
        # code向量=>ndarray
        if clock % batch_size == 0:

            # 开始时间
            time_start = time.time()

            code_vec = model(tokenizer(values, return_tensors='pt', truncation=True, padding=True)['input_ids'])[1]
            for i in range(batch_size):
                temp = code_vec[i].detach().numpy()
                code_vectors = np.append(code_vectors, code_vec[i].detach().numpy()[np.newaxis, :], axis=0)
            values = []
            clock = 0

            # 计时器
            time_end = time.time()
            time_cost = time_end - time_start
            print('batch_size:', batch_size, 'time cost', time_cost, 's')
            print('平均time cost', time_cost / batch_size, 's')

    # 去掉第一行空值
    code_vectors = np.delete(code_vectors, 0, axis=0)
    # 存code vec
    with open('code_vecs.pkl', 'wb') as vec_output_files:
        pickle.dump(code_vectors, vec_output_files)
    # 存data
    with open('data.pkl', 'wb') as data_output_files:
        pickle.dump(final_data, data_output_files)
