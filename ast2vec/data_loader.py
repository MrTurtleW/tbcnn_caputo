import os
import sys
import random

cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_path, '..'))

from config import black_ast_path, white_ast_path
from data import get_node_children
from data.parse_ast import get_ast_from_pkl

node_map = {node: index for index, node in enumerate(dir(get_node_children)) if not node.startswith('__')}


def traverse(tree):
    queue = [tree]

    while queue:
        node = queue.pop(0)
        for child in node.children:
            yield node.node_type, child.node_type
            queue.append(child)


def batch_data(batch_size):
    print('reading {}'.format(black_ast_path))
    black = get_ast_from_pkl(black_ast_path)
    print('reading {}'.format(white_ast_path))
    white = get_ast_from_pkl(white_ast_path)

    trees = black + white
    random.shuffle(trees)

    batch_X, batch_y = [], []

    for tree in trees:
        for parent, child in traverse(tree):
            if len(batch_X) == batch_size:
                yield batch_X, batch_y
                batch_X.clear()
                batch_y.clear()

            batch_X.append(node_map[child])
            batch_y.append(node_map[parent])


if __name__ == '__main__':
    for p, c in batch_data(1):
        print(p, c)
