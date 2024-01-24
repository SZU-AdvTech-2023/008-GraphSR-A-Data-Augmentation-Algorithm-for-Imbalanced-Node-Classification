import argparse

def get_args():
    """
        Description:
        Parses arguments at command line.

        Parameters:
            None

        Return:
            args - the arguments parsed
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        choices=['Cora', 'CiteSeer', 'PubMed', 'Amazon-Photo', 'Amazon-Computers', 'Coauthor-CS'],
                        default='PubMed', help='dataset name')     # Amazon-Photo 0.07 0.05   PubMed 学习率需要调整
    parser.add_argument('--mode', dest='mode', type=str, default='train')              # can be 'train' or 'test'
    parser.add_argument('--actor_model', dest='actor_model', type=str, default='')     # your actor model filename
    parser.add_argument('--critic_model', dest='critic_model', type=str, default='')   # your critic model filename
    parser.add_argument('--train_node_num', dest='train_node_num', type=int, default=20)   # number of nodes used for training
    parser.add_argument('--ratio', dest='ratio', type=float, default=0.3)   # imbalance ratio
    parser.add_argument('--isGCN', dest='isGCN', type=bool, default=True)  # can be use GCN
    args = parser.parse_args()

    return args