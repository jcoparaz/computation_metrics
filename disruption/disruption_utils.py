import math
import random
from tqdm import tqdm

def create_dict_incoming_outgoing_nodes(G):
    edges = G.edges()
    node_incoming_outgoing = {}
    for n1, n2 in tqdm(edges, desc="processed edges", total=len(edges)):
        if n1 not in node_incoming_outgoing:
            node_incoming_outgoing[n1] = {}
            node_incoming_outgoing[n1]['incoming'] = set()
            node_incoming_outgoing[n1]['outgoing'] = set()
        node_incoming_outgoing[n1]['outgoing'].add(n2)
        if n2 not in node_incoming_outgoing:
            node_incoming_outgoing[n2] = {}
            node_incoming_outgoing[n2]['incoming'] = set()
            node_incoming_outgoing[n2]['outgoing'] = set()
        node_incoming_outgoing[n2]['incoming'].add(n1)
    return node_incoming_outgoing


# also in graphs/metrics_utils.py
def sample_nodes(nodes: list, portion=0.1):
    k_length = math.ceil(portion*len(nodes))
    return random.sample(nodes, k_length)


# def main():
#     year_s, year_e, month_s, month_e = 1951, 1951, 4, 8
#     # main_folder_path = '/data/user/copara/dataset/PubMed/2022'
#     main_folder_path = '/home/cineca/Documents/dataset/PubMed/2022'#'/home/cineca/PycharmProjects/graphs/output'
#     for _y in range(year_s, year_e):
#         for _m in range(month_s, month_e):
# # /home/cineca/Documents/dataset/PubMed/2022/cit_net/1951/cit_net_1951_04.pickle
#
#     create_dict_incoming_outgoing_nodes()