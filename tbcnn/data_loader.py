import os
import sys
import pickle

from sklearn.model_selection import train_test_split

cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_path, '..'))

from config import black_ast_path, white_ast_path, embed_file
from data.parse_ast import get_ast_from_pkl, Node
from ast2vec.data_loader import node_map


class DataLoader:
    def __init__(self):
        # list of class Node
        print('reading {}'.format(black_ast_path))
        black = get_ast_from_pkl(black_ast_path)
        print('reading {}'.format(white_ast_path))
        white = get_ast_from_pkl(white_ast_path)
        asts = white + black
        labels = [0] * len(white) + [1] * len(black)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(asts, labels, test_size=0.3)

        # output of ast2vec
        with open(embed_file, 'rb') as f:
            self.embeddings = pickle.load(f)

    def batch_data(self, kind, batch_size):
        if kind == 'train':
            asts, labels = self.X_train, self.y_train
        elif kind == 'test':
            asts, labels = self.X_test, self.y_test
        else:
            raise ValueError('invalid kind {}'.format(kind))

        batch_nodes, batch_children, batch_labels = [], [], []
        for ast, label in zip(asts, labels):
            if batch_size == len(batch_nodes):
                yield self.batch_pad(batch_nodes, batch_children, batch_labels)
                batch_nodes.clear()
                batch_children.clear()
                batch_labels.clear()

            nodes, children = self.traverse(ast)
            if len(nodes) > 1:
                batch_nodes.append(nodes)
                batch_children.append(children)
                one_hot_label = [0, 0]
                one_hot_label[label] = 1
                batch_labels.append(one_hot_label)

    def traverse(self, ast):
        nodes = []
        children = []

        queue = [(ast, -1)]
        while queue:
            ast, parent_id = queue.pop(0)
            node_id = len(nodes)
            node_type = ast.node_type

            nodes.append(self.embeddings[node_map[node_type]])
            children.append([])

            if parent_id != -1:
                children[parent_id].append(node_id)

            for child in ast.children:
                queue.append((child, node_id))

        return nodes, children

    def batch_pad(self, nodes, children, labels):
        # max nodes numbers of tree in this batch
        max_nodes_num = max([len(x) for x in nodes])
        # max children number of a node in this batch
        max_children_num = max([len(child) for child in children])
        child_len = max([len(c) for n in children for c in n])

        # embedding size
        num_features = len(nodes[0][0])

        nodes = [node + [[0] * num_features] * (max_nodes_num - len(node)) for node in nodes]
        # pad batches so that every batch has the same number of nodes
        children = [child + ([[]] * (max_children_num - len(child))) for child in children]
        # pad every child sample so every node has the same number of children
        children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]

        return nodes, children, labels


if __name__ == '__main__':
    loader = DataLoader()
    for n, c, l in loader.batch_data('train', 2):
        print(n, c, l)
