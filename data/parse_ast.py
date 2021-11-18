import os
import sys
import pickle
import traceback

import js2py

cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_path, '..'))

from config import black_output_path, white_output_path, black_ast_path, white_ast_path
from data import get_node_children

sys.setrecursionlimit(30000)


class Node:
    node_type = ""

    def __init__(self, node_type):
        self.node_type = node_type
        self.children = []

    def __repr__(self):
        return self.node_type

    def __str__(self):
        return self.__repr__()


def get_ast_from_pkl(ast_path):
    asts = []
    for filename in os.listdir(ast_path):
        with open(os.path.join(ast_path, filename), 'rb') as f:
            ast = pickle.load(f)
        asts.append(ast)

    return asts


class AstParser:
    def __init__(self):
        self.esprima = js2py.require('esprima')

    def main(self):
        self.process(black_output_path, black_ast_path)
        self.process(white_output_path, white_ast_path)

    def process(self, output_path, ast_path):
        print('processing {}'.format(output_path))

        if not os.path.exists(ast_path):
            os.makedirs(ast_path)

        for filename in os.listdir(output_path):
            pkl_path = os.path.join(ast_path, filename + '.pkl')
            if os.path.exists(pkl_path):
                continue

            source = os.path.join(output_path, filename)

            try:
                with open(source, 'r') as f:
                    data = f.read().strip()
            except UnicodeDecodeError:
                traceback.print_exc()
                continue

            print('parsing {}'.format(source), end='')
            try:
                ast = self.esprima.parse(data)
                ast_converted = self.traverse(ast.to_dict())
                with open(pkl_path, 'wb') as f:
                    pickle.dump(ast_converted, f)
                print(' success')
            except js2py.internals.simplex.JsException:
                print('')
                traceback.print_exc()

    def traverse(self, ast):
        root = None
        queue = [{'ast': ast, 'parent': None}]

        while queue:
            ast = queue.pop(0)
            node_type = ast['ast']['type']
            node = Node(node_type)
            if root is None:
                root = node

            if ast['parent'] is not None:
                ast['parent'].children.append(node)

            try:
                processor = getattr(get_node_children, node_type)
            except AttributeError:
                print('got unkown type {}'.format(node_type))
                raise

            children = processor(ast['ast'])
            for child in children:
                if child is None:
                    continue
                queue.append({'ast': child, 'parent': node})

        return root


if __name__ == '__main__':
    parser = AstParser()
    parser.main()
