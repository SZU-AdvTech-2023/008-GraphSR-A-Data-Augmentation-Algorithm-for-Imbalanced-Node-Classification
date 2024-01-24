import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from model import cls_model, cls_train
import matplotlib.pyplot as plt
from arguments import get_args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def eval_policy(policy, env, test_x, test_y, features, adj_lists, labels):
    obs = env.reset()
    # done = 0
    #
    # while not done:
    #     # Query deterministic action from policy and run it
    #     action = policy(obs).detach()
    #     obs, rew, done = env.step(action)
    env.new_step()

    train_x = env.train_node
    train_y = env.train_node_y

    num_cls = np.max(labels) + 1
    count_sup = np.zeros(num_cls)
    for i in range(1, len(train_x)):
        count_sup[train_y[i]] += 1

    print('count_sup', count_sup)

    env.calculate_connect(train_x)
    classifer = cls_model(features, adj_lists, features.weight.shape[1], 128, num_cls, 'ave')
    final_train = cls_train(classifer, train_x, train_y)
    # final_train = env.cls_train(env.classifier, train_x, train_y)
    pred_test, _ = final_train.forward(test_x)


    # print("Test F1:", f1_score(test_y, pred_test.data.numpy().argmax(axis=1), average="macro"))
    one_hot = np.identity(num_cls)[test_y]

    report = classification_report(test_y.cpu(), pred_test.data.numpy().argmax(axis=1), digits=4, output_dict=True)
    print(classification_report(test_y.cpu(), pred_test.data.numpy().argmax(axis=1), digits=4))
    # 从生成的字典中提取每个类的准确率
    accuracies = {str(key): value['precision'] for key, value in report.items() if key.isdigit()}
    print(accuracies)
    auc = roc_auc_score(one_hot, pred_test.data.numpy(), average='macro')
    print("Test roc_auc:", auc)
