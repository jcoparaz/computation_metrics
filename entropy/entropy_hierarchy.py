from datetime import datetime
import pickle
import os
import networkx as nx
import math
import pandas as pd
from entropy.hierarchy import create_edges, aggregate_bottom_top

MESH_ROOT = 'MeSH'


def get_nodes_mcn(main_folder_path, year, month, sample=False):
    label_sample = '_sample' if sample else ''
    # open pickle file of monthly citation network
    path_nx_graph = '{}/{}/cit_net{}_{}_{}.pickle'.format(main_folder_path, year, label_sample, year, month)
    if not os.path.exists(path_nx_graph):
        print('{} does not exist in disk'.format(path_nx_graph))
        return
    print('{} {} opening graph'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), path_nx_graph))
    with open(path_nx_graph, "rb") as file:
        nx_graph = pickle.load(file)
    print('{} end'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), main_folder_path))
    nodes = list(nx_graph.nodes())
    return nodes


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


# entropy_hierarchy_file_path
def compute_entropy_hierarchy(main_folder_path, year, month, pmid_meshterms, tree_numbers, output_folder, sample=False):
    start_time = datetime.now()
    print('{} Computing entropy'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    nodes_mcn = get_nodes_mcn(main_folder_path, year, month, sample=sample)
    tree_node_mapping = {}
    mesh_hierarchy = nx.DiGraph()
    for pmid in nodes_mcn:
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
                # tn = tmp_tn
                meshui_treenum[mesh_desc].append(tn)
                # if tn in pruned_node:
                #     pruned_node[tn] = pruned_tn if not pruned_node[tn] else pruned_node[tn]
                # else:
                #     pruned_node[tn] = pruned_tn
                if tn not in tree_node_mapping:
                    tree_node_mapping[tn] = 1
                else:
                    tree_node_mapping[tn] += 1

        edges = create_edges(MESH_ROOT, meshui_treenum)
        mesh_hierarchy.add_edges_from(edges)
    tree_node_mapping_0 = tree_node_mapping
    # complete all missing nodes aggregating from bottom to top
    tree_node_mapping = aggregate_bottom_top(tree_node_mapping, MESH_ROOT)
    # compute entropy
    nodes_hierarchy = list(mesh_hierarchy)
    entropy = {}
    for node in nodes_hierarchy:
        prob_xi = tree_node_mapping[node]/tree_node_mapping[MESH_ROOT]
        entropy[node] = prob_xi * math.log2(1/prob_xi)

    hierarchy_mapping_initial_path = '{}/hierarchy_mapping_initial_{}_{}.csv'.format(output_folder, year, month)
    tree_node_mapping_0_pd = pd.DataFrame.from_dict(tree_node_mapping_0, orient='index', columns=['initial_count_mapping'])
    tree_node_mapping_0_pd.to_csv(hierarchy_mapping_initial_path, index=True)
    print('{} {} created'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), hierarchy_mapping_initial_path))

    hierarchy_mapping_path = '{}/hierarchy_mapping_final_{}_{}.csv'.format(output_folder, year, month)
    tree_node_mapping_pd = pd.DataFrame.from_dict(tree_node_mapping, orient='index', columns=['total_count_mapping'])
    tree_node_mapping_pd.to_csv(hierarchy_mapping_path, index=True)
    print('{} {} created'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), hierarchy_mapping_path))

    entropy_hierarchy_file_path = '{}/hierarchy_entropy_{}_{}.csv'.format(output_folder, year, month)
    tree_node_entropy_pd = pd.DataFrame.from_dict(entropy, orient='index', columns=['entropy'])
    tree_node_entropy_pd.to_csv(entropy_hierarchy_file_path, index=True)
    print('{} {} created'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), entropy_hierarchy_file_path))
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%H:%M:%S"), end_time-start_time))


if __name__== '__main__':
    pmid_meshterms_file = '/data/user/copara/code/projects/graphs/objects/pmid_meshterms.pickle'
    main_folder_path = '/data/user/copara/dataset/PubMed/2022'
    tree_numbers_file_path = '/data/user/copara/dataset/MeSH/md_treenumbers_2022.txt'
    output_folder_path = '/data/user/copara/code/projects/graphs/output'
    year_s, year_e, month_s, month_e = 2014, 2014, 1, 1

    start_time = datetime.now()
    print('{} Loading pmid_meshterms pickle'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    with open(pmid_meshterms_file, "rb") as file:
        pmid_meshterms = pickle.load(file)
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%H:%M:%S"), end_time-start_time))

    tree_numbers = load_tree_numbers(tree_numbers_file_path=tree_numbers_file_path)
    for year in range(year_s, year_e + 1):
        for month in range(month_s, month_e + 1):
            month = '0' + str(month) if len(str(month)) == 1 else month
            # entropy_hierarchy_file_path = '/data/user/copara/code/projects/graphs/output/hierarchy_entropy_{}_{}.csv'.format(year, month)
            compute_entropy_hierarchy(main_folder_path=main_folder_path, year=year, month=month, pmid_meshterms=pmid_meshterms,
                                      tree_numbers=tree_numbers, output_folder=output_folder_path)
