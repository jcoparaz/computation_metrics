from datetime import datetime
import pickle
import networkit as nk
import networkx as nx
import pandas as pd
from networkit import *
# from networkit import vizbridges
# import matplotlib.pyplot as plt

# from tqdm import tqdm
import gc
import os

def simple_example_closeness():
    # 1 2
# 10 50
# 20 30
# 20 50
# 30 40
# 40 50
# 40 60
    graph = nx.DiGraph()
    edges = {(10, 20), (10, 50), (20, 30), (20,50), (30,40), (40,50), (40,60)}
    graph.add_edges_from(edges)
    closeness_c_nx = nx.closeness_centrality(graph)
    nk_graph = nk.nxadapter.nx2nk(graph)
    closeness_c_nk = nk.centrality.Closeness(nk_graph, False, nk.centrality.ClosenessVariant.Generalized)
    closeness_c_nk.run()
    closeness_c_nk_tuples = closeness_c_nk.ranking()
    print('!!!')

def comparing_closeness_centrality(path_nx_graph, year, month):
    print(path_nx_graph)
    print("{} loading monthly citation network".format(datetime.now().strftime("%H:%M:%S")))
    with open(path_nx_graph, "rb") as file:
        nx_graph = pickle.load(file)

    num_nodes, num_edges = len(nx_graph.nodes()), len(nx_graph.edges())
    nx_graph_inverse = nx_graph.reverse()
    print('NX nodes {} edges {}'.format(num_nodes, num_edges))
    # for node in nx_graph.nodes():
    #     print(node)
    print("{} NX Computing closeness".format(datetime.now().strftime("%H:%M:%S")))
    closeness_c_nx = nx.closeness_centrality(nx_graph)

    print("{} end. Converting nx to nk graph".format(datetime.now().strftime("%H:%M:%S")))

    nk_graph = nk.nxadapter.nx2nk(nx_graph_inverse)
    num_nodes_nk, num_edges_nk = nk_graph.numberOfNodes(), nk_graph.numberOfEdges()
    print('NK nodes {} edges {}'.format(num_nodes_nk, num_edges_nk))
    print("{} end".format(datetime.now().strftime("%H:%M:%S")))

    # for (x, y) in zip(nx_graph_inverse.nodes(), range(nx_graph_inverse.number_of_nodes())):
    #     print(x, y)
    # print('==========')
    idmap = dict((id, u) for (id, u) in zip(nx_graph_inverse.nodes(), range(nx_graph_inverse.number_of_nodes())))
    idmap_inverse = dict((u, id) for (id, u) in zip(nx_graph_inverse.nodes(), range(nx_graph_inverse.number_of_nodes())))
    # for k, v in idmap.items():
    #     print(k, v)
        # if u == 20316169:
        #     print(u)
        # if u == 5667:
        #     print(u)

    print("{} NK Computing closeness".format(datetime.now().strftime("%H:%M:%S")))
    # nk.centrality.ClosenessVariant.Generalized because the graph is unconnected
    # normalized = True-> values between 0-1
    closeness_c_nk = nk.centrality.Closeness(nk_graph, True, nk.centrality.ClosenessVariant.Generalized)
    closeness_c_nk.run()
    closeness_c_nk_tuples = closeness_c_nk.ranking()
    print("{} end".format(datetime.now().strftime("%H:%M:%S")))
    print("{} Merging two centralities".format(datetime.now().strftime("%H:%M:%S")))

    both_c = []
    closeness_c_nk_values = {}
    for idx, clos in closeness_c_nk_tuples:
        if idx not in closeness_c_nk_values:
            closeness_c_nk_values[idx] = clos


    for node, clos in closeness_c_nx.items():
        both_c.append({'node': node, 'clos_nx': clos, 'clos_nk': closeness_c_nk_values[idmap[node]]})
    df = pd.DataFrame(both_c)
    # df = pd.DataFrame(closeness_c_nk_values, columns=['node', 'clos_nk'])

    # df['clos_nx'] = df['node'].map(str(closeness_c_nx))
    df.to_csv('./output/closeness_test_{}_{}.csv'.format(year, month), mode='w', sep='\t', index=None)
    print("{} end".format(datetime.now().strftime("%H:%M:%S")))

    print('!!')

def compute_closeness_nk(path_nx_graph, year, month, output_path, logging_path):
    # print(path_nx_graph)
    if not os.path.exists(path_nx_graph):
        return
    # print("{} loading monthly citation network".format(datetime.now().strftime("%H:%M:%S")))
    with open(path_nx_graph, "rb") as file:
        nx_graph = pickle.load(file)

    # no nodes in the graph
    if len(nx_graph.nodes()) == 0:
        return

    nodes_nx = nx_graph.nodes()
    num_nodes, num_edges = len(nodes_nx), len(nx_graph.edges())
    nx_graph_inverse = nx_graph.reverse()
    # print('NX nodes {} edges {}'.format(num_nodes, num_edges))

    # print("{} end. Converting nx to nk graph".format(datetime.now().strftime("%H:%M:%S")))

    nk_graph = nk.nxadapter.nx2nk(nx_graph_inverse)
    # num_nodes_nk, num_edges_nk = nk_graph.numberOfNodes(), nk_graph.numberOfEdges()
    # print('NK nodes {} edges {}'.format(num_nodes_nk, num_edges_nk))
    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))

    idmap = dict((id, u) for (id, u) in zip(nx_graph_inverse.nodes(), range(nx_graph_inverse.number_of_nodes())))

    # print("{} NK Computing closeness".format(datetime.now().strftime("%H:%M:%S")))
    # nk.centrality.ClosenessVariant.Generalized because the graph is unconnected
    # normalized = True-> values between 0-1
    closeness_c_nk = nk.centrality.Closeness(nk_graph, True, nk.centrality.ClosenessVariant.Generalized)
    closeness_c_nk.run()
    closeness_c_nk_tuples = closeness_c_nk.ranking()
    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))
    # print("{} Centralities -> dataframe -> csv".format(datetime.now().strftime("%H:%M:%S")))

    centrality_comp = []
    closeness_c_nk_values = {}
    for idx, clos in closeness_c_nk_tuples:
        if idx not in closeness_c_nk_values:
            closeness_c_nk_values[idx] = clos


    for node in nodes_nx:
        centrality_comp.append({'node': node, 'clos_nk': closeness_c_nk_values[idmap[node]]})
    df = pd.DataFrame(centrality_comp)
    # df = pd.DataFrame(closeness_c_nk_values, columns=['node', 'clos_nk'])

    # df['clos_nx'] = df['node'].map(str(closeness_c_nx))
    # df.to_csv('./output/closeness_test_{}_{}.csv'.format(year, month), mode='w', sep='\t', index=None)
    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))
    if not os.path.exists('{}closeness/'.format(output_path)):
        os.mkdir('{}closeness'.format(output_path))
    if not os.path.exists('{}closeness/{}'.format(output_path, year)):
        os.mkdir('{}closeness/{}'.format(output_path, year))
    closeness_path = '{}closeness/{}/closeness_{}_{}_new.csv'.format(output_path, year, year, month)
    df.to_csv(closeness_path, mode='w', sep='\t', index=None)
    with open(logging_path, 'a') as log_file:
        log_file.write('{} {} created\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), closeness_path))
    print('{} {} created'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), closeness_path))


def convert_dict(tuples_node_degree):
    node_degree = {}
    for node, deg in tuples_node_degree:
        if node not in node_degree:
            node_degree[node] = deg
    return node_degree

def compute_metrics_graph(path_nx_graph, year, month, output_path):
    print('{} {}'.format(datetime.now().strftime("%H:%M:%S"), path_nx_graph))
    # print("{} loading monthly citation network".format(datetime.now().strftime("%H:%M:%S")))
    with open(path_nx_graph, "rb") as file:
        nx_graph = pickle.load(file)

    # nx_graph = nx.DiGraph()
    # edges = {(10, 20), (10, 50), (20, 30), (20, 50), (30, 40), (40, 50), (40, 60)}
    # nx_graph.add_edges_from(edges)
    nodes_nx = nx_graph.nodes()
    # num_nodes, num_edges = len(nx_graph.nodes()), len(nx_graph.edges())
    # print('NX nodes {} edges {}'.format(num_nodes, num_edges))

    # print("{} Getting degree".format(datetime.now().strftime("%H:%M:%S")))
    degree = nx_graph.degree # [(20, 3), (50, 3), (10, 2), (30, 2), (40, 3), (60, 1)]
    node_degree = convert_dict(degree)
    in_degree = nx_graph.in_degree
    node_in_degree = convert_dict(in_degree)
    out_degree = nx_graph.out_degree
    node_out_degree = convert_dict(out_degree)
    all_degree = []
    for node, deg in node_degree.items():
        all_degree.append({'node': node, 'degree': deg, 'in_degree': node_in_degree[node], 'out_degree': node_out_degree[node]})
    df = pd.DataFrame(all_degree)
    if not os.path.exists('{}degree/{}'.format(output_path, year)):
        os.mkdir('{}degree/{}'.format(output_path, year))
    degree_path = '{}degree/{}/degree_{}_{}.csv'.format(output_path, year, year, month)
    df.to_csv(degree_path, mode='w', sep='\t', index=None)
    print('{} {} created'.format(datetime.now().strftime("%H:%M:%S"), degree_path))

    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))

    # print("{} Converting nx to nk graph".format(datetime.now().strftime("%H:%M:%S")))

    nk_graph = nk.nxadapter.nx2nk(nx_graph)
    num_nodes_nk, num_edges_nk = nk_graph.numberOfNodes(), nk_graph.numberOfEdges()
    # print('NK nodes {} edges {}'.format(num_nodes_nk, num_edges_nk))
    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))

    idmap = dict((id, u) for (id, u) in zip(nx_graph.nodes(), range(nx_graph.number_of_nodes())))
    del nx_graph
    # print("{} NK Computing pagerank".format(datetime.now().strftime("%H:%M:%S")))
    pagerank_c_nk = nk.centrality.PageRank(nk_graph)
    pagerank_c_nk.run()
    pagerank_c_nk_tuples = pagerank_c_nk.ranking()
    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))
    # print("{} Saving in csv file".format(datetime.now().strftime("%H:%M:%S")))
    del nk_graph
    gc.collect()

    pagerank_c = []
    pagerank_c_nk_values = {}
    for idx, prc in pagerank_c_nk_tuples:
        if idx not in pagerank_c_nk_values:
            pagerank_c_nk_values[idx] = prc

    for node in nodes_nx:
        pagerank_c.append({'node': node, 'pagerank_nk': pagerank_c_nk_values[idmap[node]]})
    df = pd.DataFrame(pagerank_c)
    # df = pd.DataFrame(closeness_c_nk_values, columns=['node', 'clos_nk'])

    # df['clos_nx'] = df['node'].map(str(closeness_c_nx))
    if not os.path.exists('{}pagerank/{}'.format(output_path, year)):
        os.mkdir('{}pagerank/{}'.format(output_path, year))
    pagerank_path = '{}pagerank/{}/pagerank_{}_{}.csv'.format(output_path, year, year, month)
    df.to_csv('{}pagerank/{}/pagerank_{}_{}.csv'.format(output_path, year, year, month), mode='w', sep='\t', index=None)
    print('{} {} created'.format(datetime.now().strftime("%H:%M:%S"), pagerank_path))


# compute degree, in-degree, out-degree, pagerank, and betweenness of nodes in graph
def compute_metrics_graph_degree_pagerank(path_nx_graph, year, month, output_path, logging_path):
    with open(logging_path, 'a') as log_file:
        log_file.write('{} {}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), path_nx_graph))
    print('{} {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), path_nx_graph))
    if not os.path.exists(path_nx_graph):
        return
    # print("{} loading monthly citation network".format(datetime.now().strftime("%H:%M:%S")))
    with open(path_nx_graph, "rb") as file:
        nx_graph = pickle.load(file)

    # no nodes in the graph
    if len(nx_graph.nodes()) == 0:
        return
    # nx_graph = nx.DiGraph()
    # edges = {(10, 20), (10, 50), (20, 30), (20, 50), (30, 40), (40, 50), (40, 60)}
    # nx_graph.add_edges_from(edges)
    nodes_nx = nx_graph.nodes()
    # num_nodes, num_edges = len(nx_graph.nodes()), len(nx_graph.edges())
    # print('NX nodes {} edges {}'.format(num_nodes, num_edges))

    # print("{} Getting degree".format(datetime.now().strftime("%H:%M:%S")))
    degree = nx_graph.degree # [(20, 3), (50, 3), (10, 2), (30, 2), (40, 3), (60, 1)]
    node_degree = convert_dict(degree)
    in_degree = nx_graph.in_degree
    node_in_degree = convert_dict(in_degree)
    out_degree = nx_graph.out_degree
    node_out_degree = convert_dict(out_degree)
    all_degree = []
    for node, deg in node_degree.items():
        all_degree.append({'node': node, 'degree': deg, 'in_degree': node_in_degree[node], 'out_degree': node_out_degree[node]})
    df = pd.DataFrame(all_degree)
    if not os.path.exists('{}degree/'.format(output_path)):
        os.mkdir('{}degree'.format(output_path))
    if not os.path.exists('{}degree/{}'.format(output_path, year)):
        os.mkdir('{}degree/{}'.format(output_path, year))
    degree_path = '{}degree/{}/degree_{}_{}.csv'.format(output_path, year, year, month)
    df.to_csv(degree_path, mode='w', sep='\t', index=None)
    with open(logging_path, 'a') as log_file:
        log_file.write('{} {} created\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), degree_path))
    print('{} {} created\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), degree_path))

    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))

    # print("{} Converting nx to nk graph".format(datetime.now().strftime("%H:%M:%S")))

    nk_graph = nk.nxadapter.nx2nk(nx_graph)
    # num_nodes_nk, num_edges_nk = nk_graph.numberOfNodes(), nk_graph.numberOfEdges()
    # print('NK nodes {} edges {}'.format(num_nodes_nk, num_edges_nk))
    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))

    idmap = dict((id, u) for (id, u) in zip(nx_graph.nodes(), range(nx_graph.number_of_nodes())))


    # print("{} NK Computing pagerank".format(datetime.now().strftime("%H:%M:%S")))
    pagerank_c_nk = nk.centrality.PageRank(nk_graph)
    pagerank_c_nk.run()
    pagerank_c_nk_tuples = pagerank_c_nk.ranking()
    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))
    # print("{} Saving in csv file".format(datetime.now().strftime("%H:%M:%S")))

    pagerank_c = []
    pagerank_c_nk_values = {}
    for idx, prc in pagerank_c_nk_tuples:
        if idx not in pagerank_c_nk_values:
            pagerank_c_nk_values[idx] = prc

    for node in nodes_nx:
        pagerank_c.append({'node': node, 'pagerank_nk': pagerank_c_nk_values[idmap[node]]})
    df = pd.DataFrame(pagerank_c)
    # df = pd.DataFrame(closeness_c_nk_values, columns=['node', 'clos_nk'])

    # df['clos_nx'] = df['node'].map(str(closeness_c_nx))
    if not os.path.exists('{}pagerank/'.format(output_path)):
        os.mkdir('{}pagerank'.format(output_path))
    if not os.path.exists('{}pagerank/{}'.format(output_path, year)):
        os.mkdir('{}pagerank/{}'.format(output_path, year))
    pagerank_path = '{}pagerank/{}/pagerank_{}_{}.csv'.format(output_path, year, year, month)
    df.to_csv(pagerank_path, mode='w', sep='\t', index=None)
    with open(logging_path, 'a') as log_file:
        log_file.write('{} {} created\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pagerank_path))
    print('{} {} created\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pagerank_path))
    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))


# compute degree, in-degree, out-degree, pagerank, and betweenness of nodes in graph
def compute_metrics_graph_betweeness(path_nx_graph, year, month, output_path, logging_path):
    with open(logging_path, 'a') as log_file:
        log_file.write('{} {}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), path_nx_graph))
    print('{} {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), path_nx_graph))
    if not os.path.exists(path_nx_graph):
        return
    # print("{} loading monthly citation network".format(datetime.now().strftime("%H:%M:%S")))
    with open(path_nx_graph, "rb") as file:
        nx_graph = pickle.load(file)

    # no nodes in the graph
    if len(nx_graph.nodes()) == 0:
        return
    # # nx_graph = nx.DiGraph()
    # # edges = {(10, 20), (10, 50), (20, 30), (20, 50), (30, 40), (40, 50), (40, 60)}
    # # nx_graph.add_edges_from(edges)
    nodes_nx = nx_graph.nodes()
    # # num_nodes, num_edges = len(nx_graph.nodes()), len(nx_graph.edges())
    # # print('NX nodes {} edges {}'.format(num_nodes, num_edges))
    #
    # # print("{} Getting degree".format(datetime.now().strftime("%H:%M:%S")))
    # degree = nx_graph.degree # [(20, 3), (50, 3), (10, 2), (30, 2), (40, 3), (60, 1)]
    # node_degree = convert_dict(degree)
    # in_degree = nx_graph.in_degree
    # node_in_degree = convert_dict(in_degree)
    # out_degree = nx_graph.out_degree
    # node_out_degree = convert_dict(out_degree)
    # all_degree = []
    # for node, deg in node_degree.items():
    #     all_degree.append({'node': node, 'degree': deg, 'in_degree': node_in_degree[node], 'out_degree': node_out_degree[node]})
    # df = pd.DataFrame(all_degree)
    # if not os.path.exists('{}degree/'.format(output_path)):
    #     os.mkdir('{}degree'.format(output_path))
    # if not os.path.exists('{}degree/{}'.format(output_path, year)):
    #     os.mkdir('{}degree/{}'.format(output_path, year))
    # degree_path = '{}degree/{}/degree_{}_{}.csv'.format(output_path, year, year, month)
    # df.to_csv(degree_path, mode='w', sep='\t', index=None)
    # with open(logging_path, 'a') as log_file:
    #     log_file.write('{} {} created'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), degree_path))
    # print('{} {} created\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), degree_path))

    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))

    # print("{} Converting nx to nk graph".format(datetime.now().strftime("%H:%M:%S")))

    nk_graph = nk.nxadapter.nx2nk(nx_graph)
    # num_nodes_nk, num_edges_nk = nk_graph.numberOfNodes(), nk_graph.numberOfEdges()
    # print('NK nodes {} edges {}'.format(num_nodes_nk, num_edges_nk))
    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))

    idmap = dict((id, u) for (id, u) in zip(nx_graph.nodes(), range(nx_graph.number_of_nodes())))


    # print("{} NK Computing pagerank".format(datetime.now().strftime("%H:%M:%S")))
    # pagerank_c_nk = nk.centrality.PageRank(nk_graph)
    # pagerank_c_nk.run()
    # pagerank_c_nk_tuples = pagerank_c_nk.ranking()
    # # print("{} end".format(datetime.now().strftime("%H:%M:%S")))
    # # print("{} Saving in csv file".format(datetime.now().strftime("%H:%M:%S")))
    #
    # pagerank_c = []
    # pagerank_c_nk_values = {}
    # for idx, prc in pagerank_c_nk_tuples:
    #     if idx not in pagerank_c_nk_values:
    #         pagerank_c_nk_values[idx] = prc
    #
    # for node in nodes_nx:
    #     pagerank_c.append({'node': node, 'pagerank_nk': pagerank_c_nk_values[idmap[node]]})
    # df = pd.DataFrame(pagerank_c)
    # # df = pd.DataFrame(closeness_c_nk_values, columns=['node', 'clos_nk'])
    #
    # # df['clos_nx'] = df['node'].map(str(closeness_c_nx))
    # if not os.path.exists('{}pagerank/'.format(output_path)):
    #     os.mkdir('{}pagerank'.format(output_path))
    # if not os.path.exists('{}pagerank/{}'.format(output_path, year)):
    #     os.mkdir('{}pagerank/{}'.format(output_path, year))
    # pagerank_path = '{}pagerank/{}/pagerank_{}_{}.csv'.format(output_path, year, year, month)
    # df.to_csv(pagerank_path, mode='w', sep='\t', index=None)
    # with open(logging_path, 'a') as log_file:
    #     log_file.write('{} {} created'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pagerank_path))
    # print('{} {} created\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pagerank_path))


    betweenness_c_nk = nk.centrality.Betweenness(nk_graph, normalized=True)
    betweenness_c_nk.run()
    betweenness_c_nk_tuples = betweenness_c_nk.ranking()
    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))
    # print("{} Setting centrality to save values".format(datetime.now().strftime("%H:%M:%S")))

    betweenness_comp = []
    betweenness_c_nk_values = {}
    for idx, betw in betweenness_c_nk_tuples:
        if idx not in betweenness_c_nk_values:
            betweenness_c_nk_values[idx] = betw


    for node in nodes_nx:
        betweenness_comp.append({'node': node, 'betw_nk': betweenness_c_nk_values[idmap[node]]})
    df = pd.DataFrame(betweenness_comp)

    if not os.path.exists('{}betweenness/'.format(output_path)):
        os.mkdir('{}betweenness'.format(output_path))
    if not os.path.exists('{}betweenness/{}'.format(output_path, year)):
        os.mkdir('{}betweenness/{}'.format(output_path, year))
    betweenness_path = '{}betweenness/{}/betweenness_{}_{}.csv'.format(output_path, year, year, month)
    df.to_csv(betweenness_path, mode='w', sep='\t', index=None)
    with open(logging_path, 'a') as log_file:
        log_file.write('{} {} created\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), betweenness_path))
    print('{} {} created'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), betweenness_path))


# compute degree, in-degree, out-degree, pagerank, and betweenness of nodes in graph
def compute_metrics_graph_v2(path_nx_graph, year, month, output_path, logging_path):
    with open(logging_path, 'a') as log_file:
        log_file.write('{} {}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), path_nx_graph))
    print('{} {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), path_nx_graph))
    if not os.path.exists(path_nx_graph):
        return
    # print("{} loading monthly citation network".format(datetime.now().strftime("%H:%M:%S")))
    with open(path_nx_graph, "rb") as file:
        nx_graph = pickle.load(file)

    # no nodes in the graph
    if len(nx_graph.nodes()) == 0:
        return
    # nx_graph = nx.DiGraph()
    # edges = {(10, 20), (10, 50), (20, 30), (20, 50), (30, 40), (40, 50), (40, 60)}
    # nx_graph.add_edges_from(edges)
    nodes_nx = nx_graph.nodes()
    # num_nodes, num_edges = len(nx_graph.nodes()), len(nx_graph.edges())
    # print('NX nodes {} edges {}'.format(num_nodes, num_edges))

    # print("{} Getting degree".format(datetime.now().strftime("%H:%M:%S")))
    degree = nx_graph.degree # [(20, 3), (50, 3), (10, 2), (30, 2), (40, 3), (60, 1)]
    node_degree = convert_dict(degree)
    in_degree = nx_graph.in_degree
    node_in_degree = convert_dict(in_degree)
    out_degree = nx_graph.out_degree
    node_out_degree = convert_dict(out_degree)
    all_degree = []
    for node, deg in node_degree.items():
        all_degree.append({'node': node, 'degree': deg, 'in_degree': node_in_degree[node], 'out_degree': node_out_degree[node]})
    df = pd.DataFrame(all_degree)
    if not os.path.exists('{}degree/'.format(output_path)):
        os.mkdir('{}degree'.format(output_path))
    if not os.path.exists('{}degree/{}'.format(output_path, year)):
        os.mkdir('{}degree/{}'.format(output_path, year))
    degree_path = '{}degree/{}/degree_{}_{}.csv'.format(output_path, year, year, month)
    df.to_csv(degree_path, mode='w', sep='\t', index=None)
    with open(logging_path, 'a') as log_file:
        log_file.write('{} {} created'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), degree_path))
    print('{} {} created\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), degree_path))

    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))

    # print("{} Converting nx to nk graph".format(datetime.now().strftime("%H:%M:%S")))

    nk_graph = nk.nxadapter.nx2nk(nx_graph)
    # num_nodes_nk, num_edges_nk = nk_graph.numberOfNodes(), nk_graph.numberOfEdges()
    # print('NK nodes {} edges {}'.format(num_nodes_nk, num_edges_nk))
    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))

    idmap = dict((id, u) for (id, u) in zip(nx_graph.nodes(), range(nx_graph.number_of_nodes())))


    # print("{} NK Computing pagerank".format(datetime.now().strftime("%H:%M:%S")))
    pagerank_c_nk = nk.centrality.PageRank(nk_graph)
    pagerank_c_nk.run()
    pagerank_c_nk_tuples = pagerank_c_nk.ranking()
    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))
    # print("{} Saving in csv file".format(datetime.now().strftime("%H:%M:%S")))

    pagerank_c = []
    pagerank_c_nk_values = {}
    for idx, prc in pagerank_c_nk_tuples:
        if idx not in pagerank_c_nk_values:
            pagerank_c_nk_values[idx] = prc

    for node in nodes_nx:
        pagerank_c.append({'node': node, 'pagerank_nk': pagerank_c_nk_values[idmap[node]]})
    df = pd.DataFrame(pagerank_c)
    # df = pd.DataFrame(closeness_c_nk_values, columns=['node', 'clos_nk'])

    # df['clos_nx'] = df['node'].map(str(closeness_c_nx))
    if not os.path.exists('{}pagerank/'.format(output_path)):
        os.mkdir('{}pagerank'.format(output_path))
    if not os.path.exists('{}pagerank/{}'.format(output_path, year)):
        os.mkdir('{}pagerank/{}'.format(output_path, year))
    pagerank_path = '{}pagerank/{}/pagerank_{}_{}.csv'.format(output_path, year, year, month)
    df.to_csv(pagerank_path, mode='w', sep='\t', index=None)
    with open(logging_path, 'a') as log_file:
        log_file.write('{} {} created'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pagerank_path))
    print('{} {} created\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pagerank_path))
    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))

    # nx_graph_inverse = nx_graph.reverse()
    # del nx_graph
    # idmap = dict((id, u) for (id, u) in zip(nx_graph_inverse.nodes(), range(nx_graph_inverse.number_of_nodes())))
    # # print("{} NK Computing closeness".format(datetime.now().strftime("%H:%M:%S")))
    #
    # # nk.centrality.ClosenessVariant.Generalized because the graph is unconnected
    # # normalized = True-> values between 0-1
    # closeness_c_nk = nk.centrality.Closeness(nk_graph, True, nk.centrality.ClosenessVariant.Generalized)
    # closeness_c_nk.run()
    # closeness_c_nk_tuples = closeness_c_nk.ranking()
    # # print("{} end".format(datetime.now().strftime("%H:%M:%S")))
    # # print("{} Centralities -> dataframe -> csv".format(datetime.now().strftime("%H:%M:%S")))
    # closeness_comp = []
    # closeness_c_nk_values = {}
    # for idx, clos in closeness_c_nk_tuples:
    #     if idx not in closeness_c_nk_values:
    #         closeness_c_nk_values[idx] = clos
    #
    # for node in nodes_nx:
    #     closeness_comp.append({'node': node, 'clos_nk': closeness_c_nk_values[idmap[node]]})
    # df = pd.DataFrame(closeness_comp)
    #
    # if not os.path.exists('{}closeness/'.format(output_path)):
    #     os.mkdir('{}closeness'.format(output_path))
    # if not os.path.exists('{}closeness/{}'.format(output_path, year)):
    #     os.mkdir('{}closeness/{}'.format(output_path, year))
    # pagerank_path = '{}closeness/{}/closeness_{}_{}.csv'.format(output_path, year, year, month)
    # df.to_csv(pagerank_path, mode='w', sep='\t', index=None)
    # print('{} {} created'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pagerank_path))

    betweenness_c_nk = nk.centrality.Betweenness(nk_graph, normalized=True)
    betweenness_c_nk.run()
    betweenness_c_nk_tuples = betweenness_c_nk.ranking()
    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))
    # print("{} Setting centrality to save values".format(datetime.now().strftime("%H:%M:%S")))

    betweenness_comp = []
    betweenness_c_nk_values = {}
    for idx, betw in betweenness_c_nk_tuples:
        if idx not in betweenness_c_nk_values:
            betweenness_c_nk_values[idx] = betw


    for node in nodes_nx:
        betweenness_comp.append({'node': node, 'betw_nk': betweenness_c_nk_values[idmap[node]]})
    df = pd.DataFrame(betweenness_comp)

    if not os.path.exists('{}betweenness/'.format(output_path)):
        os.mkdir('{}betweenness'.format(output_path))
    if not os.path.exists('{}betweenness/{}'.format(output_path, year)):
        os.mkdir('{}betweenness/{}'.format(output_path, year))
    betweenness_path = '{}betweenness/{}/betweenness_{}_{}.csv'.format(output_path, year, year, month)
    df.to_csv(betweenness_path, mode='w', sep='\t', index=None)
    with open(logging_path, 'a') as log_file:
        log_file.write('{} {} created\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), betweenness_path))
    print('{} {} created'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), betweenness_path))


def compute_metrics_missing(path_nx_graph, year, month, output_path, logging_path):
    print('{} {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), path_nx_graph))
    if not os.path.exists(path_nx_graph):
        return
    # print("{} loading monthly citation network".format(datetime.now().strftime("%H:%M:%S")))
    with open(path_nx_graph, "rb") as file:
        nx_graph = pickle.load(file)

    # no nodes in the graph
    if len(nx_graph.nodes()) == 0:
        return
    nodes_nx = nx_graph.nodes()

    nk_graph = nk.nxadapter.nx2nk(nx_graph)
    # idmap = dict((id, u) for (id, u) in zip(nx_graph_inverse.nodes(), range(nx_graph_inverse.number_of_nodes())))
    idmap = dict((id, u) for (id, u) in zip(nx_graph.nodes(), range(nx_graph.number_of_nodes())))

    with open(logging_path, 'a') as log_file:
        log_file.write('{} starting computation of betweeness\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print('{} starting computation of betweeness'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    betweenness_c_nk = nk.centrality.Betweenness(nk_graph, normalized=True)
    betweenness_c_nk.run()
    betweenness_c_nk_tuples = betweenness_c_nk.ranking()

    betweenness_comp = []
    betweenness_c_nk_values = {}
    for idx, betw in betweenness_c_nk_tuples:
        if idx not in betweenness_c_nk_values:
            betweenness_c_nk_values[idx] = betw


    for node in nodes_nx:
        betweenness_comp.append({'node': node, 'betw_nk': betweenness_c_nk_values[idmap[node]]})
    df = pd.DataFrame(betweenness_comp)

    if not os.path.exists('{}betweenness/'.format(output_path)):
        os.mkdir('{}betweenness'.format(output_path))
    if not os.path.exists('{}betweenness/{}'.format(output_path, year)):
        os.mkdir('{}betweenness/{}'.format(output_path, year))
    betweenness_path = '{}betweenness/{}/betweenness_{}_{}.csv'.format(output_path, year, year, month)
    df.to_csv(betweenness_path, mode='w', sep='\t', index=None)
    with open(logging_path, 'a') as log_file:
        log_file.write('{} {} created\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), betweenness_path))
    print('{} {} created'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), betweenness_path))


# compute selected centralities, for now it's pagerank as it can be computed for the citation networks of 2014-2021
def compute_selected_centralities(path_nx_graph, year, month, output_path):
    print('{} Pagerank {}-{}: opening nx graph and converting to nk graph'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), year, month))
    print('{} is nx graph'.format(path_nx_graph))
    with open(path_nx_graph, "rb") as file:
        nx_graph = pickle.load(file)

    # no nodes in the graph
    if len(nx_graph.nodes()) == 0:
        return
    nodes_nx = nx_graph.nodes()
    nk_graph = nk.nxadapter.nx2nk(nx_graph)

    start_time = datetime.now()
    print('{} Computing pagerank'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))

    idmap = dict((id, u) for (id, u) in zip(nx_graph.nodes(), range(nx_graph.number_of_nodes())))

    pagerank_c_nk = nk.centrality.PageRank(nk_graph)
    pagerank_c_nk.run()
    pagerank_c_nk_tuples = pagerank_c_nk.ranking()
    # print("{} end".format(datetime.now().strftime("%H:%M:%S")))
    # print("{} Saving in csv file".format(datetime.now().strftime("%H:%M:%S")))

    pagerank_c = []
    pagerank_c_nk_values = {}
    for idx, prc in pagerank_c_nk_tuples:
        if idx not in pagerank_c_nk_values:
            pagerank_c_nk_values[idx] = prc

    for node in nodes_nx:
        pagerank_c.append({'node': node, 'pagerank_nk': pagerank_c_nk_values[idmap[node]]})
    df = pd.DataFrame(pagerank_c)
    # df = pd.DataFrame(closeness_c_nk_values, columns=['node', 'clos_nk'])

    # df['clos_nx'] = df['node'].map(str(closeness_c_nx))
    if not os.path.exists('{}/pagerank'.format(output_path)):
        os.mkdir('{}/pagerank'.format(output_path))
    if not os.path.exists('{}/pagerank/{}'.format(output_path, year)):
        os.mkdir('{}/pagerank/{}'.format(output_path, year))
    pagerank_path = '{}/pagerank/{}/pagerank_{}_{}.csv'.format(output_path, year, year, month)
    df.to_csv(pagerank_path, mode='w', sep='\t', index=None)
    # with open(logging_path, 'a') as log_file:
    #     log_file.write('{} {} created'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pagerank_path))
    end_time = datetime.now()
    print('{} {} created, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), pagerank_path, end_time-start_time))


def main():
    # simple_example_closeness()
    # comparing_closeness_centrality(path_nx_graph=path_nx_graph, year=year, month=month)

    # running in Goldorak: tm_cn3
    # year, month = '2014', '01'
    # path_nx_graph='/data/user/copara/dataset/PubMed/2022/cit_net/{}/cit_net_{}_{}.pickle'.format(year, year, month)
    # compute_closeness_nk(path_nx_graph=path_nx_graph, year=year, month=month)

    # running in Goldorak
    # load monthly citation network and get degree, in_degree, out_degree.
    # compute pagerank
    # year_s, year_e, month_s, month_e = 2015, 2021, 1, 12
    # output_path = '/data/user/copara/dataset/PubMed/2022/metrics/'
    # years_range = range(year_s, year_e + 1)
    # months_range = range(month_s, month_e + 1)
    # for _year in tqdm(years_range, desc=" processed years", total=len(years_range)):
    #     for _month in tqdm(months_range, desc=" processed months", total=len(months_range)):
    #         _month = str(_month)
    #         _month = '0' + _month if len(_month) == 1 else _month
    #         path_nx_graph='/data/user/copara/dataset/PubMed/2022/cit_net/{}/cit_net_{}_{}.pickle'.format(_year, _year, _month)
    #         compute_metrics_graph(path_nx_graph, _year, _month, output_path)

    # # running in PC: 1st tab in console => ended
    # year_s, year_e, month_s, month_e = 1974, 1974, 12, 12
    # # main_path_ds = '/data/user/copara' # path Goldorak
    # main_path_ds = '/home/jennycopara/Documents' # path PC
    # output_path = '{}/dataset/PubMed/2022/metrics/'.format(main_path_ds)
    # logging_path = '{}/dataset/PubMed/2022/metrics/centralities.log'.format(main_path_ds)
    # years_range = range(year_s, year_e + 1)
    # months_range = range(month_s, month_e + 1)
    # for _year in tqdm(years_range, desc=" processed years", total=len(years_range)):
    #     for _month in tqdm(months_range, desc=" processed months", total=len(months_range)):
    #         _month = str(_month)
    #         _month = '0' + _month if len(_month) == 1 else _month
    #         path_nx_graph='{}/dataset/PubMed/2022/cit_net/{}/cit_net_{}_{}.pickle'.format(main_path_ds, _year, _year, _month)
    #         compute_metrics_missing(path_nx_graph, _year, _month, output_path, logging_path)

    # running in PC
    year_s, year_e, month_s, month_e = 1980, 2021, 1, 12
    # main_path_ds = '/data/user/copara' # path Goldorak
    main_path_ds = '/home/jennycopara/Documents' # path PC
    output_path = '{}/dataset/PubMed/2022/metrics/'.format(main_path_ds)
    logging_path = '{}/dataset/PubMed/2022/metrics/degree_pagerank.log'.format(main_path_ds)
    years_range = range(year_s, year_e + 1)
    months_range = range(month_s, month_e + 1)
    for _year in tqdm(years_range, desc=" processed years", total=len(years_range)):
        for _month in tqdm(months_range, desc=" processed months", total=len(months_range)):
            _month = str(_month)
            _month = '0' + _month if len(_month) == 1 else _month
            path_nx_graph='{}/dataset/PubMed/2022/cit_net/{}/cit_net_{}_{}.pickle'.format(main_path_ds, _year, _year, _month)
            compute_metrics_graph_degree_pagerank(path_nx_graph, _year, _month, output_path, logging_path)
            # compute_metrics_graph_v2(path_nx_graph, _year, _month, output_path, logging_path)


    # running in PC: 2nd tab in console => ended
    # year_s, year_e, month_s, month_e = 1968, 1979, 7, 12
    # # main_path_ds = '/data/user/copara' # path Goldorak
    # main_path_ds = '/home/jennycopara/Documents' # path PC
    # output_path = '{}/dataset/PubMed/2022/metrics/'.format(main_path_ds)
    # years_range = range(year_s, year_e + 1)
    # months_range = range(month_s, month_e + 1)
    # for _year in tqdm(years_range, desc=" processed years", total=len(years_range)):
    #     for _month in tqdm(months_range, desc=" processed months", total=len(months_range)):
    #         _month = str(_month)
    #         _month = '0' + _month if len(_month) == 1 else _month
    #         path_nx_graph='{}/dataset/PubMed/2022/cit_net/{}/cit_net_{}_{}.pickle'.format(main_path_ds, _year, _year, _month)
    #         compute_closeness_nk(path_nx_graph, _year, _month, output_path)
    #
    #
    # year_s, year_e, month_s, month_e = 1969, 1979, 1, 12
    # # main_path_ds = '/data/user/copara' # path Goldorak
    # main_path_ds = '/home/jennycopara/Documents' # path PC
    # output_path = '{}/dataset/PubMed/2022/metrics/'.format(main_path_ds)
    # years_range = range(year_s, year_e + 1)
    # months_range = range(month_s, month_e + 1)
    # for _year in tqdm(years_range, desc=" processed years", total=len(years_range)):
    #     for _month in tqdm(months_range, desc=" processed months", total=len(months_range)):
    #         _month = str(_month)
    #         _month = '0' + _month if len(_month) == 1 else _month
    #         path_nx_graph='{}/dataset/PubMed/2022/cit_net/{}/cit_net_{}_{}.pickle'.format(main_path_ds, _year, _year, _month)
    #         compute_closeness_nk(path_nx_graph, _year, _month, output_path)


if __name__ == '__main__':
    main()
