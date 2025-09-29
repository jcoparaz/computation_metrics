import os
import pickle
import pandas as pd
from datetime import datetime

# /home/cineca/PycharmProjects/graphs/output/di_seq_1951_04_sample.csv
# pmid_meshterms_file='/data/user/copara/code/projects/graphs/objects/pmid_meshterms_1951-1952.pickle'
def disruption_to_concepts(pmid_meshterms, disruption_file='/data/user/copara/dataset/PubMed/2022/metrics/disruption/1951/di_seq_1951_04.csv',
                           concept_di_file_path='/data/user/copara/code/projects/graphs/output/concept_disruption_1951_04.csv'):


    disruption_cn = pd.read_csv(disruption_file, names=["node", "ni", "nj", "nk", "disruption", "in", "out", "confidence"], skiprows=1)
                                # ,
                                # dtype={"node": int, "ni": int, "nj":int, "nk":int, "disruption":float, "in":int, "out":int, "confidence":float})
    disruption_cn = disruption_cn.dropna()
    # print(disruption_cn.columns)

    cols = ['node', 'disruption']
    disruption_values = disruption_cn[cols].values
    num_articles = len(disruption_cn)
    meshterms_disruption = {}
    for node, disruption in disruption_values:
        node = str(int(node))
        disruption = float(disruption)
        mesh_terms_txt = pmid_meshterms[node] if node in pmid_meshterms else ''
        if mesh_terms_txt.strip() == '':
            continue
        mesh_terms_txt = mesh_terms_txt.split()
        for mesh_descriptor in mesh_terms_txt:
            if mesh_descriptor not in meshterms_disruption:
                meshterms_disruption[mesh_descriptor] = 0
            meshterms_disruption[mesh_descriptor] += disruption
    meshterms_disruption = {mt: meshterms_disruption[mt] / num_articles for mt in meshterms_disruption}

    meshterms_disruption_pd = pd.DataFrame.from_dict(meshterms_disruption, orient='index', columns=['disruption'])
    meshterms_disruption_pd.to_csv(concept_di_file_path, index=True)
    print('{} created'.format(concept_di_file_path))
    return meshterms_disruption

import networkx as nx


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

# def load_tree_numbers(tree_numbers_file_path='/data/user/copara/dataset/MeSH/md_treenumbers_2022.txt'):
#     tree_numbers = {}
#     with open(tree_numbers_file_path) as tree_numbers_file:
#         for line in tree_numbers_file:
#             if line.strip() == '':
#                 continue
#             descriptor_numstree = line.split()
#             if len(descriptor_numstree) < 2:
#                 print('!!')
#             if descriptor_numstree[0] not in tree_numbers:
#                 tree_numbers[descriptor_numstree[0]] = descriptor_numstree[1]
#     return tree_numbers


def build_hierarchy(meshterms_disruption, tree_numbers, root):
    G = nx.DiGraph()
    # with open(_data) as data_file:
    #     data_content = data_file.read().split('\n')
    # pmid_mesh = {}
    # for line in data_content:
    #     # yyyy  mm  pmid    mesh_terms  references
    #     if line.strip() == '':
    #         continue
    #     line = line.split('\t')
    #     has_references = True if len(line)==5 else False
    #     if not has_references:
    #         continue
    #     pmid_mesh[line[2].strip()] = line[3].strip()


    tree_node_dis = {}
    for mesh_term, disruption in meshterms_disruption.items():
        meshui_treenum = {}
        curr_tree_numbers = tree_numbers[mesh_term].split(' ') if mesh_term in tree_numbers else ['']
        meshui_treenum[mesh_term] = []
        for tree_num in curr_tree_numbers:
            tree_num = tree_num.strip()
            # some unique ids do not have tree number, e.g., D005260
            if tree_num == '':
                continue
            meshui_treenum[mesh_term].append(tree_num)
            if tree_num not in tree_node_dis:
                tree_node_dis[tree_num] = disruption
            else:
                tree_node_dis[tree_num] += disruption

        edges = create_edges(root, meshui_treenum)
        G.add_edges_from(edges)
    return G, tree_node_dis


def get_parent(tree_node):
    index_point = tree_node.rfind('.')
    # if tree_node contains a point, e.g., 'B01.050.500.131.617.720.500.500.750.712.500.875.225'
    if index_point > -1:
        parent = tree_node[0:index_point]
    else:
    # e.g. 'B01'
        parent = tree_node[0]
    return parent


def compute_disruption_tree_nodes(G, tree_node_dis, tree_node_dis_file_path, disruption_hierarchy_file_path):
    tree_node_dis_pd = pd.DataFrame.from_dict(tree_node_dis, orient='index', columns=['disruption'])
    tree_node_dis_pd.to_csv(tree_node_dis_file_path, index=True)
    print('{} created'.format(tree_node_dis_file_path))

    treenode_depth = nx.single_source_shortest_path_length(G, 'MeSH')
    # depth_treenodes contains tree nodes by level, e.g., 1:{A, B, C, ...}, 2:{A01, A11, B01, ...}
    depth_treenodes = {}
    for node, depth in treenode_depth.items():
        if depth not in depth_treenodes:
            depth_treenodes[depth] = set()
        depth_treenodes[depth].add(node)
    # deepest_level = len(depth_treenodes) - 1
    disruption_hierarchy = {}
    # current_level = deepest_level
    list_levels_reverse = sorted(depth_treenodes.keys(), reverse=True)
    for current_level in list_levels_reverse:
        unique_parents = set()
        # get parents of all treenodes at this level
        for treenode in depth_treenodes[current_level]:
            parent = get_parent(treenode)
            unique_parents.add(parent)

        for parent in unique_parents:
            # if parent == 'B01.050.150.900.649.313.988.400.112.199.120':
            #     print('!!!')
            # apply equation
            children = set(G[parent])
            di_parent = 0
            for child in children:
                # if child in tree_node_dis means that MeSH descriptors, e.g.D0010, have mapped to the tree node child
                if child in tree_node_dis:
                    # if child does not have children then add it to disruption_hierarchy
                    # otherwise the disruptiveness of this tree node depends on children also
                    if len(list(G[child])) == 0:
                        disruption_hierarchy[child] = tree_node_dis[child]
                # disruption_hierarchy[child] to use the latest disruption value computed so far
                di_parent += disruption_hierarchy[child]
            di_parent = di_parent/len(depth_treenodes[current_level])
            if parent not in disruption_hierarchy:
                # direct_disruptiveness is the disruption value if 'parent' node already has disruption
                direct_disruptiveness = tree_node_dis[parent] if parent in tree_node_dis else 0
                disruption_hierarchy[parent] = di_parent + direct_disruptiveness

    disruption_hierarchy_pd = pd.DataFrame.from_dict(disruption_hierarchy, orient='index', columns=['disruption'])
    disruption_hierarchy_pd.to_csv(disruption_hierarchy_file_path, index=True)
    print('{} created'.format(disruption_hierarchy_file_path))
    # print('!!!')

def aggregate_bottom_top(tree_node_value: dict, root: str, num_concepts: int, treenode_disruption_file_path: str):
    new_tree_node_value = tree_node_value.copy()
    for node, value in tree_node_value.items():
        num_levels = node.count('.')
        parent_node = node
        for i in range(0, num_levels):
            index_point = parent_node.rfind('.')
            parent_node = parent_node[0:index_point]
            if parent_node in new_tree_node_value:
                new_tree_node_value[parent_node] += value
            else:
                new_tree_node_value[parent_node] = value
        # to aggregate level 2 (A01, A10, A11, B01, B02...) to level 1 (A, B)
        if parent_node[0] in new_tree_node_value:
            new_tree_node_value[parent_node[0]] += value
        else:
            new_tree_node_value[parent_node[0]] = value
    # to include the root in the aggregation: A, B, C, ... counts will be sum into MeSH node
    new_tree_node_value[root] = 0
    for node, acc in new_tree_node_value.items():
        if len(node) == 1:
            # if root not in new_tree_node_value:
            new_tree_node_value[root] += new_tree_node_value[node]


    new_tree_node_value_disruption = {tn: new_tree_node_value[tn] / num_concepts for tn in new_tree_node_value}

    new_tree_node_value_disruption_pd = pd.DataFrame.from_dict(new_tree_node_value_disruption, orient='index', columns=['disruption'])
    new_tree_node_value_disruption_pd.to_csv(treenode_disruption_file_path, index=True)
    print('{} created'.format(treenode_disruption_file_path))

    return new_tree_node_value

# '../data/2000_01_short.txt'
def prune_treenumber(tree_number, level):
    '''

    :param level:
        -1: do not prune the tree
        >=1: prune at this level
    :return:
        string portion of the tree_number at n level
        whether the string was pruned or not: True or False
    '''
    # MeSH                  --> level 0: root of hierarchy,
    # A B   C   D ...       --> level 1: classes
    # A10 A11 D02 ...       --> level 2
    # A10.165 E05.225 ...   --> level 3

    # C01.150.252.289.225.250
    # C01.150.252.400.210.125.245
    # C01.375.354.220.250
    # C11.187.183.220.250
    # C11.294.354.220.250
    if not tree_number[0].isalpha():
        return
    if level == -1:
        return tree_number, False
    elif level == 1:
        pruned = True if len(tree_number) > 1 else False
        return tree_number[0], pruned
    else:
        # points = find_occurrences(tree_number, '.')
        # current_level = level-1
        # if len(points) <= current_level:
        #     return tree_number
        # return tree_number[0:points[current_level]]
        parts = tree_number.split('.')
        num_blocks = len(parts)
        if num_blocks < level:
            return tree_number, False
        return '.'.join(parts[0:level-1]), True


def propagate_disruption_cn_hierarchy():
    # disruption_file = '/data/user/copara/dataset/PubMed/2022/metrics/disruption/1951/di_seq_1951_04.csv',
    pmid_meshterms_file = '/data/user/copara/code/projects/graphs/objects/pmid_meshterms.pickle'
    root = 'MeSH'
    # concept_di_file_path = '/data/user/copara/code/projects/graphs/output/concept_disruption_1951_04.csv'
    year_s, year_e, month_s, month_e = 1951, 1951, 5, 8
    sample=False

    sample_lbl = '_sample' if sample else ''

    start_time = datetime.now()
    print('{} Loading pmid_meshterms pickle'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    with open(pmid_meshterms_file, "rb") as file:
        pmid_meshterms = pickle.load(file)
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%H:%M:%S"), end_time-start_time))

    for year in range(year_s, year_e + 1):
        for month in range(month_s, month_e + 1):
            month = str(month)
            month = '0' + month if len(month) == 1 else month
            # disruption_file = '/data/user/copara/dataset/PubMed/2022/metrics/disruption/{}/di_seq_{}_{}{}.csv'.format(year, year, month, sample_lbl)
            disruption_file = '/data/user/copara/code/projects/graphs/output/di_seq_dict_{}_{}{}.csv'.format(year, month, sample_lbl)
            concept_di_file_path = '/data/user/copara/code/projects/graphs/output/concept_disruption_{}_{}{}.csv'.format(year, month, sample_lbl)
            treenode_di_initial_file_path = '/data/user/copara/code/projects/graphs/output/treenode_disruption_initial_{}_{}{}.csv'.format(year, month, sample_lbl)
            treenode_di_file_path = '/data/user/copara/code/projects/graphs/output/treenode_disruption_{}_{}{}.csv'.format(year, month, sample_lbl)
            disruption_concepts = disruption_to_concepts(pmid_meshterms=pmid_meshterms, disruption_file=disruption_file,
                                                         concept_di_file_path=concept_di_file_path)
            tree_numbers = load_tree_numbers()
            graph_hierarchy, tree_node_disruption = build_hierarchy(meshterms_disruption=disruption_concepts,
                                                                    tree_numbers=tree_numbers, root=root)
            compute_disruption_tree_nodes(G=graph_hierarchy, tree_node_dis=tree_node_disruption,
                                          tree_node_dis_file_path=treenode_di_initial_file_path,
                                          disruption_hierarchy_file_path=treenode_di_file_path)
            # aggregated_tree_nodes = aggregate_bottom_top(tree_node_value=tree_node_disruption, root=root,
            #                                              num_concepts=len(disruption_concepts),
            #                                              treenode_disruption_file_path=treenode_di_file_path)
            # print('!!!')


def propagate_disruption_cn_hierarchy_sample():
    # disruption_file = '/data/user/copara/dataset/PubMed/2022/metrics/disruption/1951/di_seq_1951_04.csv',
    main_folder_disruption = '/home/cineca/Documents/dataset/PubMed/2022/metrics_sample/disruption'
    #/home/cineca/Documents/dataset/PubMed/2022/metrics_sample/disruption/2014/di_mp_dict_2014_01.csv
    # pmid_meshterms_file = '/data/user/copara/code/projects/graphs/objects/pmid_meshterms.pickle' # G
    pmid_meshterms_file = '/home/cineca/PycharmProjects/graphs/objects/pmid_meshterms.pickle' # laptop
    treenumbers_path = '/home/cineca/Documents/dataset/MeSH_descriptor_data/md_treenumbers_2022.txt'

    root = 'MeSH'
    # concept_di_file_path = '/data/user/copara/code/projects/graphs/output/concept_disruption_1951_04.csv'
    year_s, year_e, month_s, month_e = 2014, 2021, 1, 12
    # sample=True
    #
    # sample_lbl = '_sample' if sample else ''

    start_time = datetime.now()
    print('{} Loading pmid_meshterms pickle'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    with open(pmid_meshterms_file, "rb") as file:
        pmid_meshterms = pickle.load(file)
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%H:%M:%S"), end_time-start_time))

    for year in range(year_s, year_e + 1):
        for month in range(month_s, month_e + 1):
            start_time = datetime.now()
            month = str(month)
            month = '0' + month if len(month) == 1 else month
            folder_disruption_year = '{}/{}'.format(main_folder_disruption, year)
            # disruption_file = '/data/user/copara/dataset/PubMed/2022/metrics/disruption/{}/di_seq_{}_{}{}.csv'.format(year, year, month, sample_lbl)
            disruption_file = '{}/di_mp_dict_{}_{}.csv'.format(folder_disruption_year, year, month)
            concept_di_file_path = '{}/concept_disruption_{}_{}.csv'.format(folder_disruption_year, year, month)
            treenode_di_initial_file_path = '{}/treenode_disruption_initial_{}_{}.csv'.format(folder_disruption_year, year, month)
            treenode_di_file_path = '{}/treenode_disruption_{}_{}.csv'.format(folder_disruption_year, year, month)
            disruption_concepts = disruption_to_concepts(pmid_meshterms=pmid_meshterms, disruption_file=disruption_file,
                                                         concept_di_file_path=concept_di_file_path)
            tree_numbers = load_tree_numbers(tree_numbers_file_path=treenumbers_path)
            graph_hierarchy, tree_node_disruption = build_hierarchy(meshterms_disruption=disruption_concepts,
                                                                    tree_numbers=tree_numbers, root=root)
            compute_disruption_tree_nodes(G=graph_hierarchy, tree_node_dis=tree_node_disruption,
                                          tree_node_dis_file_path=treenode_di_initial_file_path,
                                          disruption_hierarchy_file_path=treenode_di_file_path)
            # aggregated_tree_nodes = aggregate_bottom_top(tree_node_value=tree_node_disruption, root=root,
            #                                              num_concepts=len(disruption_concepts),
            #                                              treenode_disruption_file_path=treenode_di_file_path)
            # print('!!!')
            end_time = datetime.now()
            print('{} {}-{} lasted {}'.format(end_time.strftime("%H:%M:%S"), year, month, end_time - start_time))


def propagate_disruption_cn_hierarchy_sample_process(source_folder_disruption, destination_folder_disruption,
                                                             year, month, pmid_meshterms_year, tree_numbers_year):

    # sample=True
    #
    # sample_lbl = '_sample' if sample else ''
    root = 'MeSH'
    # start_time = datetime.now()
    # print('{} Loading pmid_meshterms pickle'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    # with open(pmid_meshterms_file, "rb") as file:
    #     pmid_meshterms = pickle.load(file)
    # end_time = datetime.now()
    # print('{} end, lasted {}'.format(end_time.strftime("%H:%M:%S"), end_time-start_time))

    # for year in range(year_s, year_e + 1):
    #     for month in range(month_s, month_e + 1):
    #         start_time = datetime.now()
    #         month = str(month)
    #         month = '0' + month if len(month) == 1 else month
    start_time = datetime.now()
    # folder_disruption_year = '{}/{}'.format(main_folder_disruption, year)
    source_folder_disruption_year = '{}/{}'.format(source_folder_disruption, year)
    destination_folder_disruption_year = '{}/{}'.format(destination_folder_disruption, year)
    if not os.path.exists(destination_folder_disruption_year):
        os.makedirs(destination_folder_disruption_year)
    disruption_file = '{}/di_mp_dict_{}_{}.csv'.format(source_folder_disruption_year, year, month)
    concept_di_file_path = '{}/concept_disruption_{}_{}.csv'.format(destination_folder_disruption_year, year, month)
    treenode_di_initial_file_path = '{}/treenode_disruption_initial_{}_{}.csv'.format(destination_folder_disruption_year, year, month)
    treenode_di_file_path = '{}/treenode_disruption_{}_{}.csv'.format(destination_folder_disruption_year, year, month)
    disruption_concepts = disruption_to_concepts(pmid_meshterms=pmid_meshterms_year, disruption_file=disruption_file,
                                                 concept_di_file_path=concept_di_file_path)
    # tree_numbers = load_tree_numbers(tree_numbers_file_path=tree_numbers_year)
    graph_hierarchy, tree_node_disruption = build_hierarchy(meshterms_disruption=disruption_concepts,
                                                            tree_numbers=tree_numbers_year, root=root)
    compute_disruption_tree_nodes(G=graph_hierarchy, tree_node_dis=tree_node_disruption,
                                  tree_node_dis_file_path=treenode_di_initial_file_path,
                                  disruption_hierarchy_file_path=treenode_di_file_path)
    # aggregated_tree_nodes = aggregate_bottom_top(tree_node_value=tree_node_disruption, root=root,
    #                                              num_concepts=len(disruption_concepts),
    #                                              treenode_disruption_file_path=treenode_di_file_path)
    # print('!!!')
    end_time = datetime.now()
    print('{} {}-{} lasted {}'.format(end_time.strftime("%H:%M:%S"), year, month, end_time - start_time))

def propagate_disruption_cn_hierarchy_sample_pmyears():
    pubmed_folder = '../data'
    source_folder_disruption = '../output/metrics_sample/disruption'
    destination_folder_disruption = '../output/metrics_sample/disruption'

    year_s, year_e, month_s, month_e = 2014, 2014, 1, 2
    for year in range(year_s, year_e + 1):
        start_time = datetime.now()
        pmid_meshterms_year_file = '{}/objects/{}_pmid_meshterms.pickle'.format(pubmed_folder, year)
        tree_numbers_year_file_path = '{}/objects/md_treenumbers_{}.txt'.format(pubmed_folder, year)
        print('{} Loading pmid_meshterms pickle'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
        with open(pmid_meshterms_year_file, "rb") as file:
            pmid_meshterms_year = pickle.load(file)
        end_time = datetime.now()
        print('{} end, lasted {}'.format(end_time.strftime("%H:%M:%S"), end_time - start_time))
        tree_numbers_year = load_tree_numbers(tree_numbers_file_path=tree_numbers_year_file_path)
        for month in range(month_s, month_e + 1):
            month = '0' + str(month) if len(str(month)) == 1 else month
            print('--------------{}-{}--------------'.format(year, month))
            propagate_disruption_cn_hierarchy_sample_process(source_folder_disruption, destination_folder_disruption,
                                                             year, month, pmid_meshterms_year, tree_numbers_year)

if __name__ == '__main__':
    # propagate_disruption_cn_hierarchy()
    # propagate_disruption_cn_hierarchy_sample()
    propagate_disruption_cn_hierarchy_sample_pmyears()