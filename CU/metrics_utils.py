import os
import pickle
from datetime import datetime

import math
import random

random.seed(10)

def load_tree_numbers(tree_numbers_file_path='/data/user/copara/dataset/MeSH/md_treenumbers_2022.txt'):
    # load tree_numbers of MeSH
    # UID: tree_numbers separated by space
    tree_numbers = {}
    # tree_num_file: str = './data/tree_numbers.txt'
    with open(tree_numbers_file_path, 'rt') as tree_file:
        tree_content = tree_file.read().split('\n')
    for line in tree_content:
        if line.strip() == '':
            continue
        line = line.split('\t')
        tree_numbers[line[0].strip()] = line[1].strip()
    return tree_numbers


def get_nodes_mcn(main_folder_path, year, month, prefix_filename):
    # open pickle file of monthly citation network
    path_nx_graph = '{}/{}/{}_{}_{}.pickle'.format(main_folder_path, year, prefix_filename, year, month)
    if not os.path.exists(path_nx_graph):
        print('{} does not exist in disk'.format(path_nx_graph))
        return
    print('{} {} opening graph'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), path_nx_graph))
    with open(path_nx_graph, "rb") as file:
        nx_graph = pickle.load(file)
    print('{} end'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), main_folder_path))
    nodes = list(nx_graph.nodes())
    return nodes


def create_edges(root, meshui_treenum):
    all_edges = []
    for ui in meshui_treenum:
        # line = line.split('\t')
        tree_numbers = meshui_treenum[ui]
        edges = []
        for t in tree_numbers:
            if t.strip() == '':
                continue
            edges.extend(get_edges(t, root))
        all_edges.extend(edges)
    return all_edges

# M01.060.057
def get_edges(tree_number, root):
    edges = []
    edges.append((root, tree_number[0]))
    edges.append((tree_number[0], tree_number[0: 3]))
    # cut_tree_number = tree_number[1:]
    parts = tree_number.split('.')
    lbl = ''
    for i in range(0, len(parts)-1):
        lbl += parts[i] + '.'
        edges.append((lbl.strip('.'), lbl + parts[i+1]))
    return edges


def sample_nodes(nodes: list, portion=0.1):
    k_length = math.ceil(portion*len(nodes))
    return random.sample(nodes, k_length)