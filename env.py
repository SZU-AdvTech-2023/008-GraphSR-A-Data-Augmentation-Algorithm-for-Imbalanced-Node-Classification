import math

import numpy as np
import random
import torch
from sklearn import metrics
from collections import deque

from sklearn.metrics import classification_report

from model import Encoder, MeanAggregator, SupervisedGraphSage
from collections import defaultdict
import torch.nn.functional as F
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Env:
    def __init__(self, node_label, node_label_y, node_unlabel, val_x, val_y, features, adj_lists, test_x, test_y,
                 num_cls, args, all_label):
        self.args = args
        self.all_labels = all_label
        self.val_x = val_x
        self.val_y = val_y
        self.train_node_ori = node_label.clone().detach()
        self.train_node_y_ori = node_label_y.clone().detach()
        self.test_x = test_x
        self.test_y = test_y
        self.num_cls = num_cls  # 类别数
        self.adj_lists = adj_lists
        self.action_space = 2
        self.classifier = self.cls_model(features, adj_lists, features.weight.shape[1], 128, 'weight')
        self.unlabel_set = node_unlabel
        self.candidate_node = []
        self.candidate_node_y = []
        self.pred_can = []
        self.emb_can = []
        self.pred_v = []
        self.past_performance = deque(maxlen=10)
        # self.class_less = [4, 5, 6]
        # self.count_tr_ratio = [20, 20, 20, 20, 6, 6, 6]

        self.count_tr_ratio, self.class_less = self.ratio_less()
        print(self.class_less, self.count_tr_ratio)
        self.count = 0
        self.done = 0
        self.emb_sum = torch.zeros((128, num_cls))
        self.emb_cen = torch.zeros((128, num_cls))
        self.mean = 0
        self.f1 = 0
        self.f1_max = 0

    def ratio_less(self):
        split_loc = math.ceil(self.num_cls / 2)
        count_tr_ratio = []
        class_less = []
        for i in range(self.num_cls):
            if i < split_loc:
                count_tr_ratio.append(self.args.train_node_num)
            else:
                count_tr_ratio.append(int(self.args.train_node_num * self.args.ratio))
        # count_tr_ratio = np.array(count_tr_ratio)
        for k in range(0, self.num_cls):  # 修改了
            class_less.append(k)
        return count_tr_ratio, class_less

    def reset(self):
        self.count = 0
        self.done = 0
        self.train_node = self.train_node_ori
        self.train_node_y = self.train_node_y_ori
        self.supplement_emb = torch.zeros(128)
        # self.past_performance = torch.zeros(10)
        self.past_performance = deque(maxlen=10)
        # self.mean = 0

        self.calculate_connect(self.train_node)

        pre_train = self.cls_train(100, self.classifier, self.train_node, self.train_node_y)  # epoch = 50

        # print("self.train", self.train_node.shape)
        pred_l, emb_l = pre_train.forward(self.train_node)

        # print('1pred_l', pred_l, emb_l)
        pred_y1 = pred_l.data.numpy().argmax(axis=1)
        # print('pre_y1', pred_y1, np.unique(pred_y1))

        self.emb_l = torch.sum(emb_l.detach(), 1)  # [128, 98] 按行求和 -> 128

        pred_u, emb_u = pre_train.forward(self.unlabel_set)

        # print('1pred_u', pred_u)
        # print('1emb_u', emb_u)
        # print('1emb_l', emb_l)

        self.candidate_node, self.candidate_node_y = self.SelectNode(emb_l, pred_u, emb_u)

        self.pred_can, self.emb_can = pre_train.forward(self.candidate_node)
        self.pred_v, _ = pre_train.forward(self.val_x)
        self.f1 = metrics.f1_score(self.val_y, self.pred_v.detach().numpy().argmax(axis=1), average="macro")
        self.f1_max = self.f1
        self.past_performance.append(self.f1)

        obs = torch.cat((self.emb_l, self.supplement_emb.detach()), -1)
        obs = torch.cat((obs, self.emb_can[:, self.count].detach()), -1)

        return obs

    def get_nei_node(self, k):
        nei = set([])
        for i, node in enumerate(self.train_node):
            if self.train_node_y[i] == k:
                nei = nei.union(set(self.adj_lists.get(int(node))))

        return nei

    def new_step(self):

        split_loc = math.ceil(self.num_cls / 2)

        retrain = self.cls_train(50, self.classifier, self.train_node, self.train_node_y)  # 10
        self.pred_v, _ = retrain.forward(self.val_x)

        report = classification_report(self.val_y, self.pred_v.detach().numpy().argmax(axis=1), digits=4,
                                       output_dict=True)
        # 从生成的字典中提取每个类的准确率
        accuracies = {str(key): value['precision'] for key, value in report.items() if key.isdigit()}
        print(accuracies)
        # 使用sorted函数对字典的项进行排序，并取出前两个最小项
        sorted_items = sorted(accuracies.items(), key=lambda x: x[1])[:1]

        # 从排序后的元组中提取键
        keys = [int(item[0]) for item in sorted_items]
        print(keys)
        print(len(self.candidate_node), len(self.candidate_node_y))
        # 需要加入的数量
        # add_node_num_flag = [15, 0, 0, 0, 25, 15, 15]
        # add_node_num_flag = [0, 0, 0, 25, 15, 15]
        add_node_num_flag = [0, 10, 20]
        self.train_node = list(self.train_node)
        self.train_node_y = list(self.train_node_y)
        for k in range(self.num_cls):
            k = int(k)
            nei_nodes = self.get_nei_node(k)
            add_node_num = 0
            for i, y in enumerate(self.candidate_node_y):  # 添加节点
                if add_node_num >= add_node_num_flag[k]:
                    break
                if k == y and self.candidate_node[i] in nei_nodes:
                    self.train_node.append(self.candidate_node[i])
                    self.train_node_y.append(self.all_labels[self.candidate_node[i]][0])
                    add_node_num += 1
            for i, y in enumerate(self.candidate_node_y):  # 添加节点
                if add_node_num >= add_node_num_flag[k]:
                    break
                if k == y:
                    self.train_node.append(self.candidate_node[i])
                    self.train_node_y.append(self.all_labels[self.candidate_node[i]][0])
                    add_node_num += 1

    def step(self, action_ori):
        action = torch.argmax(F.softmax(action_ori, dim=-1))

        current_node = torch.cat((self.train_node, torch.tensor([self.candidate_node[self.count]])), 0)
        current_y = torch.cat((self.train_node_y, torch.tensor([self.candidate_node_y[self.count]])), 0)
        # print('current', current_y.shape, current_node.shape)
        # if (self.count+1)%10 ==
        retrain = self.cls_train(10, self.classifier, current_node, current_y)  # 10
        self.pred_v, _ = retrain.forward(self.val_x)
        f1_score = metrics.f1_score(self.val_y, self.pred_v.detach().numpy().argmax(axis=1), average="macro")
        self.mean = np.mean(list(self.past_performance))

        if action == 1:
            self.supplement_emb = self.supplement_emb + self.emb_can[:, self.count].detach()
            self.train_node = current_node
            self.train_node_y = current_y

            if f1_score - self.mean > 0:
                reward = 1
            elif f1_score - self.mean == 0:
                reward = 0
            else:
                reward = -1

            self.f1 = f1_score

        else:

            if f1_score - self.mean > 0:
                reward = -1
            elif f1_score - self.mean == 0:
                reward = 0
            else:
                reward = 1

        reward = torch.tensor(reward)

        self.count += 1
        self.past_performance.append(self.f1)
        # print('len', len(self.candidate_node))
        if self.count == len(self.candidate_node):
            # print('len', len(self.candidate_node), self.count)
            self.done = 1
            self.count = 0
            self.supplement_emb = torch.zeros(128)

        obs = torch.cat((self.emb_l, self.supplement_emb), -1)
        self.state = torch.cat((obs, self.emb_can[:, self.count].detach()), -1)

        return self.state, reward, self.done

    def SelectNode(self, emb_l, pred_u, emb_u):  # choose unlabel nodes based on distance
        # calculate centroids of clusters
        # print('num_cls', self.num_cls)
        # print('pred_u', pred_u)
        # print('emb_u', emb_u)
        # print('emb_l', emb_l)
        emb_sum = torch.zeros((128, self.num_cls))

        for i in range(len(self.train_node)):
            emb_sum[:, self.train_node_y[i]] = emb_sum[:, self.train_node_y[i]] + emb_l[:, i]
        emd_cen = torch.zeros((128, self.num_cls))
        for i in range(emb_sum.shape[1]):
            emd_cen[:, i] = emb_sum[:, i] / self.count_tr_ratio[i]

        dict_node = defaultdict(list)
        dict_unemb = defaultdict(list)

        pred_y = pred_u.data.numpy().argmax(axis=1)
        unique, counts = np.unique(pred_y, return_counts=True)
        result = dict(zip(unique, counts))
        print('pre_y', result)

        for i in range(len(self.unlabel_set)):
            dict_node[pred_y[i]].append(self.unlabel_set[i])
            dict_unemb[pred_y[i]].append(emb_u[:, i].detach().numpy())
        node_num = 80
        # print(dict_node, dict_unemb)
        # split_loc = int(self.num_cls/2)

        c = 0
        supplement = np.zeros(self.num_cls * node_num, dtype=int)
        supplement_y = np.zeros(self.num_cls * node_num, dtype=int)

        # 创建节点编号和节点标签的字典
        node_label_dict = {}
        # print('dict', dict_node)
        for i in self.class_less:
            cen = emd_cen[:, i].detach().numpy()  # shape=128
            # print('dict', i, dict_node.get(i))
            node = np.array(dict_node.get(i))
            emb = np.array(dict_unemb.get(i))

            dis = []

            selnodes = node[0:node_num]

            for j in range(len(node)):
                distance = np.linalg.norm(cen - emb[j])
                if j < node_num:
                    dis.append(distance)
                else:
                    dis_max = max(dis)
                    idx_max = dis.index(dis_max)
                    if distance < dis_max:
                        dis[idx_max] = distance
                        selnodes[idx_max] = node[j]

            dis_node = zip(dis, selnodes)
            dis_node_sort = sorted(dis_node, key=lambda x: x[0])
            dis_sort, selnodes_sort = [list(x) for x in zip(*dis_node_sort)]
            p = 0
            for x in range(len(selnodes)):
                supplement[p + c] = selnodes_sort[x]
                supplement_y[p + c] = i
                p += self.num_cls
            c += 1
        #     for x in range(len(selnodes)):
        #         node_label_dict.update({selnodes_sort[x]: i})
        #
        # # 将字典转换为对应的x和y数组
        # x = np.array(list(node_label_dict.keys()))
        # y = np.array(list(node_label_dict.values()))
        # 随机打乱顺序
        # permutation = np.random.permutation(len(x))
        # x = x[permutation]
        # y = y[permutation]

        return supplement, supplement_y

    def cls_model(self, features, adj_lists, fea_size, hidden, loss_fun):
        isGCN = self.args.isGCN
        agg1 = MeanAggregator(features, cuda=True)
        enc1 = Encoder(features, fea_size, hidden, adj_lists, agg1, gcn=isGCN, cuda=False)
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, hidden, adj_lists, agg2,
                       base_model=enc1, gcn=isGCN, cuda=False)
        enc1.num_samples = 5
        enc2.num_samples = 5
        graphsage = SupervisedGraphSage(self.num_cls, enc2, loss_fun)

        return graphsage

    def cls_train(self, epoch, graphsage, train_x, train_y):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)  # 0.7
        # train_y = labels[np.array(train)].squeeze()
        for batch in range(epoch):
            batch_nodes = train_x[:128]
            batch_y = train_y[:128]
            # random.shuffle(train)
            c = list(zip(train_x, train_y))
            random.shuffle(c)
            train_x, train_y = zip(*c)

            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, batch_y)
            # with torch.autograd.detect_anomaly():
            #     loss.backward()
            loss.backward()

            # for name, param in graphsage.named_parameters():
            #     if param.grad is not None and torch.isnan(param.grad).any():
            #         print("nan gradient found")
            #         print("name:", name)
            #         print("param:", param.grad)
            #         raise SystemExit

            optimizer.step()

        return graphsage

    def calculate_connect(self, nodes):  # 计算节点间的连接数
        nodes = np.array(nodes)
        nodes = set(nodes)
        edges_num = 0
        cross_edges_num = 0
        for node in nodes:
            edges = self.adj_lists[int(node)]
            edges_num += len(edges)
            cross_edges_num += len((set(edges) & nodes))
        print('nodes num', len(nodes), 'cross_edges_num', cross_edges_num, 'edges_num', edges_num)
