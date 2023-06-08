import numpy as np
from data_preprocessor import *
import ast
import heapq

data_name_list = ['Ml1M', 'Ml100k', 'Hetrec-ML', 'Yahoo']
num_users_list = [6040, 943, 2113, 15400]
num_items_list = [3952, 1682, 10109, 1000]

model_name = "MMA"

top_k_list = [5, 10]

for i in range(len(data_name_list)):
    sample_data = []
    user_ids = []
    data_name = data_name_list[i]
    num_users = num_users_list[i]
    num_items = num_items_list[i]

    train_ratio = 0.9
    path = "data_7_1_2/%s" % data_name + "/"
    result_path = "results/" + model_name + "/%s" % data_name + "/"
    with open(path + "test_samples.txt", 'r') as file:
        for line in file:
            line = line.strip()
            item = ast.literal_eval(line)
            sample_data.append(item)

    with open(path + "test_samples_user_id.txt", 'r') as file:
        for line in file:
            line = line.strip()
            item = ast.literal_eval(line)
            user_ids.append(item)

    out_mat = read_decoder(result_path, num_users, num_items)

    for top_k in top_k_list:

        total_ndcg = 0
        total_hit = 0
        test_users = 0
        for u in range(len(user_ids)):
            u_id = user_ids[u]
            sample = sample_data[u]
            ndcg = 0
            hit = 0
            if len(sample[0][1]) + len(sample) < 11:
                continue
            test_users += 1
            for s in range(len(sample)):
                pos_id = sample[s][0]
                neg_list = sample[s][1]
                neg_list.insert(0, pos_id)

                predict_value = [out_mat[u_id][i] for i in neg_list]
                top_k_idx = heapq.nlargest(top_k, range(len(predict_value)), key=predict_value.__getitem__)

                if 0 in top_k_idx:
                    target_idx = top_k_idx.index(0)
                    hit += 1
                    ndcg += 1 / (np.log2(target_idx + 1 + 1))

            total_ndcg += ndcg / len(sample)
            total_hit += hit / len(sample)

        print("ndcg@" + str(top_k) + ":" + str(total_ndcg / test_users),
              "hit@" + str(top_k) + ":" + str(total_hit / test_users))

