import copy
import math

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score, roc_auc_score
from collections import defaultdict
from sklearn import manifold
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid

from model import Encoder, MeanAggregator, SupervisedGraphSage, MLP

from eval_policy import eval_policy
import sys
from env import Env
from arguments import get_args
from collections import defaultdict
import scipy.sparse as sp


def extract_indices(label_array, k, label):
    indices = [i for i in range(len(label_array)) if label_array[i] == label]
    return random.sample(indices, k) if len(indices) >= k else indices


def data_spilit(labels, num_cls, args):
    num_nodes = labels.shape[0]
    split_loc = math.ceil(num_cls/2)
    count_tr_ratio = []
    for i in range(num_cls):
        if i < split_loc:
            count_tr_ratio.append(args.train_node_num)
        else:
            count_tr_ratio.append(int(args.train_node_num*args.ratio))
    count_tr_ratio = np.array(count_tr_ratio)
    print(count_tr_ratio)

    extract_labels = copy.copy(labels)
    # count_tr_ratio = np.array([20, 20, 20, 20, 6, 6, 6])
    train_set = []
    for l, k in enumerate(count_tr_ratio):
        extracted_indices = extract_indices(extract_labels, k, l)
        train_set.extend(extracted_indices)

    for i in train_set:
        extract_labels[i] = -1

    test = []
    for l in range(num_cls):
        extracted_indices = extract_indices(extract_labels, 100, l)
        test.extend(extracted_indices)

    for i in test:
        extract_labels[i] = -1

    val = []
    for l in range(num_cls):
        extracted_indices = extract_indices(extract_labels, 30, l)
        val.extend(extracted_indices)

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
    train_y = np.array(train_y)
    print(np.sum(train_y == 0), np.sum(train_y == 1), np.sum(train_y == 2))
    print('len', len(train_set), len(val), len(test), len(unlable))

    return train_set, train_y, val, val_y, test, test_y, unlable


def data_spilit111(labels, num_cls, args):
    num_nodes = labels.shape[0]
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1500]
    val = rand_indices[1500:2000]
    train_set = list(rand_indices[2000:])
    # train = random.sample(train, 100)

    tr_ratio = []
    count_tr = np.zeros(num_cls)
    # count_tr_ratio = np.array([20, 6, 20, 6, 20, 20, 6])
    count_tr_ratio = np.array([20, 20, 20, 20, 6, 6, 6])
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

def load_data(path):
    data = torch.load(path)
    # print(data)
    x = data[0]['x']
    edge_index = data[0]['edge_index']
    y = data[0]['y']

    x = np.array(x)
    y = np.array(y)
    y = y.reshape(len(y), 1)
    edge_index = np.array(edge_index)

    nodes = [i for i in range(y.shape[0])]

    graph = defaultdict(set)  # 创建一个 defaultdict，值的类型为 set

    for i, node in enumerate(nodes):
        graph[i].add(i)

    edge_index = np.array(edge_index)
    start_nodes = edge_index[0]
    connected_nodes = edge_index[1]

    for start, end in zip(start_nodes, connected_nodes):
        graph[start].add(end)
        graph[end].add(start)

    print(x.shape, y.shape)
    unique, counts = np.unique(y, return_counts=True)

    result = dict(zip(unique, counts))
    print(result)
    return x, y, graph

def run_cora(args):
    np.random.seed(1)
    random.seed(1)


    file_path = 'dataset/' + args.dataset +'.pt'
    feat_data, labels, adj_lists = load_data(file_path)
    # print('feat_data', feat_data)
    features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()\
    num_cls = np.max(labels) + 1
    train_x, train_y, val_x, val_y, test_x, test_y, unlable = data_spilit(labels, num_cls, args)

    train_x = torch.LongTensor(train_x)
    train_y = torch.LongTensor(train_y)
    val_x = torch.LongTensor(val_x)
    val_y = torch.LongTensor(val_y)
    test_x = torch.LongTensor(test_x)
    test_y = torch.LongTensor(test_y)
    unlable = torch.LongTensor(unlable)


    return train_x, train_y, val_x, val_y, test_x, test_y, unlable, features, adj_lists, labels, num_cls


def test_model(environment, actor_model, test_x, test_y, features, adj_lists, labels):
    """
        Tests the model.

        Parameters:
            env - the environment to test the policy on
            actor_model - the actor model to load in

        Return:
            None
    """

    # If the actor model is not specified, then exit
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = 384  # env.observation_space.shape[0]
    act_dim = 2  # env.action_space.shape[0]

    # Build our policy the same way we build our actor model in PPO
    policy = MLP(obs_dim, act_dim)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

    eval_policy(policy=policy, env=environment, test_x=test_x, test_y=test_y, features=features, adj_lists=adj_lists,
                labels=labels)


def main(args):
    """
        The main function to run.

        Parameters:
            args - the arguments parsed from command line

        Return:
            None
    """

    train_x, train_y, val_x, val_y, test_x, test_y, unlable, features, adj_lists, labels, num_cls = run_cora(args)

    environment = Env(train_x, train_y, unlable, val_x, val_y, features, adj_lists, test_x, test_y, num_cls, args, labels)

    if args.mode == 'test':
        test_model(environment=environment, actor_model='ppo_actor.pth', test_x=test_x, test_y=test_y,
                   features=features,
                   adj_lists=adj_lists, labels=labels)


if __name__ == "__main__":
    args = get_args()  # Parse arguments from command line
    args.mode = "test"
    print("===start===")
    main(args)
