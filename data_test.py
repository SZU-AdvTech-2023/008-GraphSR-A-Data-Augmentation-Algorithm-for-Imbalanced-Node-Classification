import random

import numpy as np
import torch
from torch import nn
from collections import defaultdict
np.set_printoptions(threshold=np.inf)

def data_spilit(labels, num_cls):
    num_nodes = labels.shape[0]
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1500]
    val = rand_indices[1500:2000]
    train_set = list(rand_indices[2000:])
    # train = random.sample(train, 100)

    tr_ratio = {'Cora': 3, 'CiteSeer': 3, 'PubMed': 2, 'Amazon-Photo': 2, 'Amazon-Computers': 2, 'Coauthor-CS': 2}
    count_tr = np.zeros(num_cls)
    # count_tr_ratio = np.array([20, 6, 20, 6, 20, 20, 6])
    count_tr_ratio = np.array([20, 20, 20, 20, 20, 20, 20])
    for i in train_set:
        for j in range(num_cls):
            if labels[i] == j:
                count_tr[j] += 1
                break
        # if count_tr[labels[i]] <= 20:
        #     tr_balanced.append(i)
        # count_tr[labels[i]] += 1
        if count_tr[labels[i]] <= count_tr_ratio[labels[i]]:
            tr_ratio.append(i)
    train_set = tr_ratio

    test_balanced = []
    count_test = np.zeros(num_cls)
    for i in test:
        for j in range(num_cls):
            if labels[i] == j:
                count_test[j] += 1
                break
        if count_test[labels[i]] <= 100:
            test_balanced.append(i)
    test = test_balanced

    val_bal = []
    count_val = np.zeros(num_cls)
    for i in val:
        for j in range(num_cls):
            if labels[i] == j:
                count_val[j] += 1
                break
        if count_val[labels[i]] <= 30:
            val_bal.append(i)
    val = val_bal

    index = np.arange(0, num_nodes)
    unlable = np.setdiff1d(index, train_set)
    unlable = np.setdiff1d(unlable, val)
    unlable = np.setdiff1d(unlable, test)
    # train_x = train
    train_y = []
    for i in train_set:
        train_y.append(int(labels[i]))
    # print(train_y)
    val_y = []
    for i in val:
        val_y.append(int(labels[i]))
    test_y = []
    for i in test:
        test_y.append(int(labels[i]))

    return train_set, train_y, val, val_y, test, test_y, unlable


def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_cls = 7
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()

    train_x, train_y, val_x, val_y, test_x, test_y, unlable = data_spilit(labels, num_cls)

    train_x = torch.LongTensor(train_x)
    train_y = torch.LongTensor(train_y)
    val_x = torch.LongTensor(val_x)
    val_y = torch.LongTensor(val_y)
    test_x = torch.LongTensor(test_x)
    test_y = torch.LongTensor(test_y)
    unlable = torch.LongTensor(unlable)

    return train_x, train_y, val_x, val_y, test_x, test_y, unlable, features, adj_lists, labels, feat_data

load_cora()
train_x, train_y, val_x, val_y, test_x, test_y, unlable, features, adj_lists, labels, feat_data = run_cora()
feat_data = torch.Tensor(feat_data)
labels = [int(x) for x in labels]
print(labels)
labels = torch.Tensor(labels).long()
labels = labels.squeeze()
print(train_x.shape, train_y.shape)
print(val_x.shape, val_y.shape)
print(test_x.shape, test_y.shape, feat_data.shape)
print(unlable.shape)
print(labels)
# print(adj_lists)


# print(adj_lists)
# labels = torch.argmax(label_pkl, -1)
# print(labels)
# all_data = torch.tensor(all_data)


row_arr = []
col_arr = []
value_arr = [1, 2, 3]

for i in adj_lists:
    r = i
    col = adj_lists[i]
    for c in col:
        row_arr.append(r)
        col_arr.append(c)

print(len(row_arr), len(col_arr))
csr_len = len(row_arr)
value_arr = [1 for x in range(csr_len)]

import scipy.sparse as sp
# 定义一个空的稀疏矩阵

adj_s = sp.csr_matrix((value_arr, (row_arr, col_arr)), shape=[2708, 2708])

# print(adj_s)
# 上面的解释所建立的方式
all_data = []
all_data.append(adj_s)
all_data.append(feat_data)
all_data.append(labels)
all_data.append(train_x)
all_data.append(val_x)
all_data.append(test_x)


torch.save(all_data, './cora/cora_new.pt')
print("finished")