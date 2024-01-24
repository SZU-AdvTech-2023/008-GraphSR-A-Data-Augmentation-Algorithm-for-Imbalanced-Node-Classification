from collections import defaultdict

import numpy as np
from torch_geometric.datasets import Planetoid
import networkx as nx
import matplotlib.pyplot as plt

data_citeseer = Planetoid(root='./citeseer/', name='citeseer')
x = data_citeseer.x
edge_index = data_citeseer.edge_index
y = data_citeseer.y
print(x.shape)
print(edge_index.shape)
print(y.shape)


print(edge_index)
x = np.array(x)
delete_index = []
num2 = 0
for i, elem in enumerate(x):
    num1 = np.sum(elem == 0)
    if num1 == x.shape[1]:
        delete_index.append(i)
        num2 += 1
print(num2)

nodes = [i for i in range(y.shape[0])]
nodes = np.delete(nodes, delete_index)
node_map = {}

graph = defaultdict(set)  # 创建一个 defaultdict，值的类型为 set

G = nx.Graph()
split = 300

for i in range(split):
    G.add_node(i)

for i, node in enumerate(nodes):
    node_map[node] = i
    graph[i].add(i)
    # if i < split:
        # G.add_edge(i, i)
edge_index = np.array(edge_index)
start_nodes = edge_index[0]
connected_nodes = edge_index[1]


print('len', len(start_nodes))
for start, end in zip(start_nodes, connected_nodes):
    if start in delete_index or end in delete_index:
        continue
    paper1 = node_map[start]
    paper2 = node_map[end]
    graph[paper1].add(paper2)
    graph[paper2].add(paper1)
    if paper1 < split and paper2 < split:
        G.add_edge(paper1, paper2)

x = np.delete(x, delete_index, 0)
num2 = 0
for i, elem in enumerate(x):
    num1 = np.sum(elem == 0)
    if num1 == 0:
        delete_index.append(i)
        num2 += 1
print(num2)




y = np.delete(y, delete_index)
print(x.shape, len(y), len(graph))
y = np.array(y)
print(x)

num_s = 0
for i in range(len(y)):
    nodes = graph.get(i)
    nodes = [x for x in nodes]
    labels = np.array(y[[nodes]])
    k = np.sum(labels == y[i])
    # print(y[i], labels)
    if k == len(labels):
        num_s += 1

print(num_s, num_s/len(y))


# 分配颜色
unique_labels = list(set(y[:split]))
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
label_color_map = dict(zip(unique_labels, colors))
node_colors = [label_color_map[label] for label in y[:split]]
pos = nx.random_layout(G)
node_colors = np.array(node_colors)
print(node_colors.shape)
# 绘制图
nx.draw(G, pos=pos, with_labels=False, node_color=node_colors, node_size=70)
plt.show()



