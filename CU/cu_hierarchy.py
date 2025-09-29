from datetime import datetime
import pickle
import CU.metrics_utils as metrics_utils
import numpy as np
import scipy.sparse as sp
import networkx as nx
# from copy import deepcopy
import pandas as pd

MESH_ROOT = 'MeSH'

def get_parents(tree_node):
    parents = []
    parent_node = tree_node
    index_point = parent_node.rfind('.')
    while index_point >= 0:
        parent_node = parent_node[0:index_point]
        parents.append(parent_node)
        index_point = parent_node.rfind('.')
    else:
        # to aggregate level 2 (A01, A10, A11, B01, B02...) to level 1 (A, B)
        parents.append(parent_node[0])
    print('!!!')
    return parents


def get_temporary_matrix(selected_idx, id_to_node, node_to_id, pmid_to_id, node_pmids, cols)->sp.dok_matrix:
    # temporary dictionary containing only the selected indexes
    tmp_dict = {}
    for idx in selected_idx:
        tmp_dict[id_to_node[idx]] = node_pmids[id_to_node[idx]]
    # equivalence of indexes in temporary matrix wrt original indexes
    new_idxs = dict((idx, i) for i, idx in enumerate(selected_idx))
    # temporary matrix of size (number_selected_idx, original_number_cols)
    tmp_matrix = sp.dok_matrix((len(selected_idx), cols), dtype=np.bool_)
    for tree_node, pmids in tmp_dict.items():
        for pmid in pmids:
            id_node = node_to_id[tree_node]
            tmp_matrix[new_idxs[id_node], pmid_to_id[pmid]] = 1
    return tmp_matrix


def compute_cu(main_folder_path, year, month, pmid_meshterms, tree_numbers, output_path, prefix_input_filename='cit_net', prefix_output_filename='hierarchy_cu'):
    pmids_mcn = metrics_utils.get_nodes_mcn(main_folder_path, year, month, prefix_input_filename)
    start_time_cu = datetime.now()
    output_cu_computation = ''
    output_cu_computation += ('{} computing CU\n'.format(start_time_cu.strftime("%Y-%m-%d %H:%M:%S")))
    # hierarchy_nodes: key is the tree node in the hierarchy, value is a list of papers that map the tree node
    hierarchy_nodes = {}
    # mesh_hierarchy contain the hierarchy for this month
    mesh_hierarchy = nx.DiGraph()
    for pmid in pmids_mcn:
        # when the pmid is not in pmid_meshterms
        # e.g., 33607720 is reference of 6 papers but does not exist in PubMed anymore
        if str(pmid) not in pmid_meshterms:
            continue
        mesh_terms = pmid_meshterms[str(pmid)].strip()
        # some pmid does not have MeSH terms
        if mesh_terms == '':
            continue
        mesh_descriptors = mesh_terms.split(' ') # papers with no mesh_terms?
        meshui_treenum = {}
        for mesh_desc in mesh_descriptors:
            if mesh_desc not in tree_numbers:
                continue
            mesh_desc_tree_numbers = tree_numbers[mesh_desc]
            # some unique ids do not have tree number, e.g., D005260
            if mesh_desc_tree_numbers == '':
                continue
            tree_nodes = mesh_desc_tree_numbers.split(' ') # mesh descriptor without tree node?

            meshui_treenum[mesh_desc] = []
            for tn in tree_nodes:
                tn = tn.strip()
                meshui_treenum[mesh_desc].append(tn)
                if tn not in hierarchy_nodes:
                    hierarchy_nodes[tn] = []
                hierarchy_nodes[tn].append(pmid)
        edges = metrics_utils.create_edges(MESH_ROOT, meshui_treenum)
        mesh_hierarchy.add_edges_from(edges)
    hierarchy_nodes = dict(sorted(hierarchy_nodes.items()))
    rows, cols = len(hierarchy_nodes), len(pmids_mcn)
    matrix_features = sp.dok_matrix((rows, cols), dtype=np.bool_)
    id_to_node = dict((i, n) for i, n in enumerate(hierarchy_nodes.keys()))
    id_to_pmid = dict((i, n) for i, n in enumerate(pmids_mcn))
    node_to_id = dict((n, i) for i, n in enumerate(hierarchy_nodes.keys()))
    pmid_to_id = dict((n, i) for i, n in enumerate(pmids_mcn))
    for tree_node, pmids in hierarchy_nodes.items():
        for pmid in pmids:
            matrix_features[node_to_id[tree_node], pmid_to_id[pmid]] = 1
    # add nodes in hierarchy which were not mapped, but they are real nodes in the hierarchy
    # these additional nodes are intermediate nodes
    # hierarchy_nodes_with_intermediate = list(hierarchy_nodes.keys())
    # # hierarchy_nodes_with_intermediate.append('MeSH')
    # for node in hierarchy_nodes.keys():
    #     parents = get_parents(node)
    #     for par in parents:
    #         if par not in hierarchy_nodes_with_intermediate:
    #             hierarchy_nodes_with_intermediate.append(par)
    treenode_depth = nx.single_source_shortest_path_length(mesh_hierarchy, MESH_ROOT)
    # depth_treenodes contains tree nodes by level, e.g., 1:{A, B, C, ...}, 2:{A01, A11, B01, ...}, 3:{'C08.381', 'A08.340', 'A04.531', ...}
    depth_treenodes = {}
    for node, depth in treenode_depth.items():
        if depth not in depth_treenodes:
            depth_treenodes[depth] = set() #[]
        depth_treenodes[depth].add(node)#.append(node)

    total_mappings = matrix_features.nnz
    # sum of columns
    total_per_feature = matrix_features.sum(axis=0)
    p_fk = total_per_feature / total_mappings
    treenode_cu = {}
    num_nodes_by_levels = []
    # runtime = []
    # for level, tree_nodes in depth_treenodes.items():
    for level in sorted(depth_treenodes.keys(), reverse=True):
        tree_nodes = depth_treenodes[level]
        output_cu_computation += ('Level {} number of nodes {}\n'.format(level, len(tree_nodes)))
        num_nodes_by_levels.append(len(tree_nodes))
        # level, tree_nodes = 4, ['A01.378.610', 'A01.378.800']
        for treenode in tree_nodes: # depth_treenodes[4]
        # for i, treenode in enumerate(tree_nodes):

            # if i > 1500:
            #     continue
            # start_time = datetime.now()

            # CU for tree node A01.378.610
            descendants = list(nx.descendants(mesh_hierarchy, treenode))
            # add parent as it can have a mapping to it
            descendants.append(treenode)
            descendants.sort()

            index_desdendants = []
            for descendant in descendants:
                if descendant in node_to_id:
                    # features_node = vstack([features_node, matrix_features.getrow(node_to_id[descendant])])
                    # row = matrix_features[node_to_id[descendant]]
                    # features_node = vstack([features_node, row])
                    index_desdendants.append(node_to_id[descendant])
            # selected_idx = np.array(index_desdendants)
            # features_node = matrix_features[selected_idx, :]
            if level == 0:
                # means that only tree node is MeSH node and features are the whole matrix_features

                # features_node = matrix_features
                p_fk_c = p_fk
                total_mappings_category = total_mappings
            else:
                features_node = get_temporary_matrix(selected_idx=index_desdendants, id_to_node=id_to_node, node_to_id=node_to_id,
                                                     pmid_to_id=pmid_to_id, node_pmids=hierarchy_nodes, cols=cols)
                # total_mappings_category = features_node.sum()
                total_mappings_category = features_node.nnz

                # numerator: sum of columns of the descendant nodes with features
                p_fk_c = features_node.sum(axis=0) / total_mappings_category
            p_c = total_mappings_category / total_mappings
            diff_pfkc_pfk = np.power(p_fk_c, 2) - np.power(p_fk, 2)
            treenode_cu[treenode] = p_c * np.matrix.sum(diff_pfkc_pfk)
            # end_time = datetime.now()
            # runtime.append(end_time-start_time)
            # if i % 1000 == 0:
            #     print('{} Level {} {} out of {}'.format(datetime.now(), level, i, len(depth_treenodes[level])))
    num_nodes_by_levels.reverse()
    output_cu_computation += ' '.join(map(str, num_nodes_by_levels))# (num_nodes_by_levels)
    hierarchy_cu_path = '{}/{}_{}_{}.csv'.format(output_path, prefix_output_filename, year, month)
    hierarchy_cu_pd = pd.DataFrame.from_dict(treenode_cu, orient='index', columns=['cu'])
    hierarchy_cu_pd.to_csv(hierarchy_cu_path, index=True)
    output_cu_computation += ('\n{} {} created\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), hierarchy_cu_path))
    end_time_cu = datetime.now()
    output_cu_computation += ('CU computation lasted {}\n'.format(end_time_cu-start_time_cu))

    # print('Average time of processing per tree node ', pd.to_timedelta(pd.Series(runtime)).mean())
    # print('!!!')
    return output_cu_computation

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

def main():
    # Goldorak
    pmid_meshterms_file = '/data/user/copara/code/projects/graphs/objects/pmid_meshterms.pickle'
    tree_numbers_file_path = '/data/user/copara/dataset/MeSH/md_treenumbers_2022.txt'
    main_folder_path = '/data/user/copara/dataset/PubMed/2022/cit_net'
    output_path = '/data/user/copara/code/projects/graphs/output'
    prefix_input_filename, prefix_output_filename = 'cit_net', 'hierarchy_cu'

    # pmid_meshterms_file = '/data/user/copara/code/projects/graphs/objects/pmid_meshterms.pickle'
    # tree_numbers_file_path = '/data/user/copara/dataset/MeSH/md_treenumbers_2022.txt'
    # main_folder_path = '/data/user/copara/dataset/PubMed/2022/cit_net_sample'
    # output_path = '/data/user/copara/code/projects/graphs/output'
    # prefix_input_filename, prefix_output_filename = 'cit_net_sample', 'hierarchy_cu_sample'

    # /data/user/copara/dataset/PubMed/2022/cit_net_sample/2021

    # laptop
    # pmid_meshterms_file = '/home/cineca/PycharmProjects/graphs/objects/pmid_meshterms.pickle'
    # tree_numbers_file_path = '/home/cineca/Documents/dataset/MeSH_descriptor_data/md_treenumbers_2022.txt'
    # main_folder_path = '/home/cineca/Documents/dataset/PubMed/2022'
    # output_path = '/home/cineca/PycharmProjects/graphs/output'

    start_time = datetime.now()
    print('{} Loading pmid_meshterms pickle'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    with open(pmid_meshterms_file, "rb") as file:
        pmid_meshterms = pickle.load(file)
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%H:%M:%S"), end_time-start_time))

    tree_numbers = metrics_utils.load_tree_numbers(tree_numbers_file_path=tree_numbers_file_path)

    year_s, year_e, month_s, month_e = 1951, 1951, 4, 8
    for year in range(year_s, year_e + 1):
        for month in range(month_s, month_e + 1):
            month = '0' + str(month) if len(str(month)) == 1 else month
            compute_cu(main_folder_path, year, month, pmid_meshterms, tree_numbers, output_path,
                       prefix_input_filename=prefix_input_filename, prefix_output_filename=prefix_output_filename)


if __name__ == '__main__':
    main()