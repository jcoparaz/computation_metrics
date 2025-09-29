import logging
# from disruption import disruption_utils

import disruption.disruption_utils # S
# import disruption.disruption_utils # G-> disruption.disruption_utils # Baobab  import disruption_utils
import networkx as nx
import numpy as np
import tqdm
from tqdm import tqdm
import pandas as pd
import os
import pickle
from datetime import datetime
import multiprocessing
import math
import warnings
# import disruption_mcn
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.options.mode.chained_assignment = None

PRIOR = 10
mp = multiprocessing.Manager()
_disruption_shared_var = mp.list()
_posterior_shared_var = mp.list()


def rank_nodes(disrupt):
    # print('Computing disruption posterior')
    disrupt = disrupt.dropna()
    cols = ['ni', 'nj', 'nk', 'disruption']
    # diffs = []
    confidences = []
    disruption_values = disrupt[cols].values
    for ni, nj, nk, disruption in tqdm(disruption_values, desc="dis-posterior", total=len(disruption_values)):
        D = np.random.dirichlet([PRIOR + ni,
                                 PRIOR + nj,
                                 PRIOR + nk], size=10000)
        # pos_i = D[:, 0]
        # pos_j = D[:, 1]
        if disruption <= 0:
            confidence = ((D[:, 0] - D[:, 1]) < 0).mean()
        else:
            confidence = ((D[:, 0] - D[:, 1]) > 0).mean()
        confidences.append(confidence)
        # diff = pos_i - pos_j
        # diffs.append(diff)

    disrupt['confidence'] = confidences
    # posteriors = pd.DataFrame(diffs, index=disrupt.index)
    # posteriors['name'] = disrupt['name']
    # posteriors['confidence'] = confidences
    return disrupt#, posteriors

# from parallelbar import progress_map
# from multiprocessing import current_process


def worker_disruption_indices(data_slice_ini, data_slice_end, min_in, min_out, proc):
    # np.savetxt("data_slice_{}.csv".format(proc), data_slice, delimiter=",")
    # pd.DataFrame(data_slice).to_csv("data_slice_{}.csv".format(proc))
    range_data = range(data_slice_ini, data_slice_end)

    # current = current_process()
    # for node_id in tqdm(range_data, desc="disruption-{}".format(current.name), total=len(range_data), position=current._identity[0]-1):
    for node_id in range_data:
        if in_count[id_to_node[node_id]] >= min_in and \
                out_count[id_to_node[node_id]] >= min_out:
            ni = 0
            nj = 0
            nk = 0

            # outgoing = F[node_id].nonzero()[1]
            outgoing = node_incoming_outgoing[id_to_node[node_id]]['outgoing']
            # incoming = T[:, node_id].nonzero()[0]
            incoming = node_incoming_outgoing[id_to_node[node_id]]['incoming']
            outgoing_set = set(outgoing)

            for other_id in incoming:
                # second_level = F[other_id].nonzero()[1]
                second_level = node_incoming_outgoing[other_id]['outgoing']
                if len(outgoing_set.intersection(second_level)) == 0:
                    ni += 1
                else:
                    nj += 1

            # who mentions my influences
            # who_mentions_my_influences = np.unique(T[:, outgoing].nonzero()[0])
            who_mentions_my_influences = set()
            for outg in outgoing:
                who_mentions_my_influences = who_mentions_my_influences.union(node_incoming_outgoing[outg]['incoming'])
            # who_mentions_my_influences = np.unique(node_incoming_outgoing[id_to_node[outgoing]]['incoming'])
            for other_id in who_mentions_my_influences:
                # do they mention me?! if no, add nk
                # if F[other_id, node_id] == 0 and other_id != node_id:
                if F[node_to_id[other_id], node_id] == 0 and node_to_id[other_id] != node_id:
                    nk += 1

            _disruption_shared_var.append({'node': id_to_node[node_id], 'ni': ni, 'nj': nj, 'nk': nk, 'disruption': (ni - nj) / (ni + nj + nk),
                               'in': in_count[id_to_node[node_id]], 'out': out_count[id_to_node[node_id]]})

        else:
            _disruption_shared_var.append({'node': id_to_node[node_id], 'ni': np.nan, 'nj': np.nan, 'nk': np.nan, 'disruption': np.nan,
                               'in': in_count[id_to_node[node_id]], 'out': out_count[id_to_node[node_id]]})
            # D[node_id, 0] = id_to_node[node_id]
            # D[node_id, 1] = np.nan
            # D[node_id, 2] = np.nan
            # D[node_id, 3] = np.nan
            # D[node_id, 4] = np.nan
            # D[node_id, 5] = in_count[id_to_node[node_id]]
            # D[node_id, 6] = out_count[id_to_node[node_id]]
    # with open("shared_var_{}.csv".format(proc), 'w') as output_file:
    #     for _line in _disruption_shared_var:
    #         output_file.write('{}\n'.format(_line))
    #     tqdm.update(1)

    return

def worker_disruption(data_slice, min_in, min_out, proc, log_file_path, debug=False):
    # np.savetxt("data_slice_{}.csv".format(proc), data_slice, delimiter=",")
    # pd.DataFrame(data_slice).to_csv("data_slice_{}.csv".format(proc))
    # range_data = range(data_slice_ini, data_slice_end)
    # data_slice = nodes[data_slice_ini: data_slice_end]

    # current = current_process()
    # for node_id in tqdm(range_data, desc="disruption-{}".format(current.name), total=len(range_data), position=current._identity[0]-1):
    # for node_id in range_data:
    total_data = len(data_slice)
    for i, node_id in enumerate(data_slice):
        try:
            # with open(log_file_path, 'a') as log_file:
            #     log_file.write('core {} i {} node_id {} in_count[node_id] {} out_count[node_id] {}'.format(proc, i, node_id, in_count[node_id], out_count[node_id]))
            if in_count[node_id] >= min_in and \
                    out_count[node_id] >= min_out:
                ni = 0
                nj = 0
                nk = 0

                # outgoing = F[node_id].nonzero()[1]
                outgoing = node_incoming_outgoing[node_id]['outgoing']
                # incoming = T[:, node_id].nonzero()[0]
                incoming = node_incoming_outgoing[node_id]['incoming']
                outgoing_set = set(outgoing)

                # with open(log_file_path, 'a') as log_file:
                #     log_file.write('core {} i {} node_id {} outgoing {}'.format(proc, i, node_id, outgoing))
                #     log_file.write('core {} i {} node_id {} incoming {}'.format(proc, i, node_id, incoming))

                for other_id in incoming:
                    # second_level = F[other_id].nonzero()[1]
                    second_level = node_incoming_outgoing[other_id]['outgoing']
                    if len(outgoing_set.intersection(second_level)) == 0:
                        ni += 1
                    else:
                        nj += 1
                # with open(log_file_path, 'a') as log_file:
                #     log_file.write('core {} i {} node_id {} ni {} nj {}'.format(proc, i, node_id, ni, nj))

                # who mentions my influences
                # who_mentions_my_influences = np.unique(T[:, outgoing].nonzero()[0])
                who_mentions_my_influences = set()
                for outg in outgoing:
                    who_mentions_my_influences = who_mentions_my_influences.union(node_incoming_outgoing[outg]['incoming'])
                # who_mentions_my_influences = np.unique(node_incoming_outgoing[id_to_node[outgoing]]['incoming'])
                for other_id in who_mentions_my_influences:
                    # do they mention me?! if no, add nk
                    # if F[other_id, node_id] == 0 and other_id != node_id:

                    # if F[node_to_id[other_id], node_id] == 0 and node_to_id[other_id] != node_id:
                    #     nk += 1

                    if other_id != node_id and node_id not in node_incoming_outgoing[other_id]["outgoing"]:
                        nk += 1
                # with open(log_file_path, 'a') as log_file:
                #     log_file.write('core {} i {} node_id {} nk {}\n'.format(proc, i, node_id, nk))

                _disruption_shared_var.append({'node': node_id, 'ni': ni, 'nj': nj, 'nk': nk, 'disruption': (ni - nj) / (ni + nj + nk),
                                   'in': in_count[node_id], 'out': out_count[node_id]})

            else:
                _disruption_shared_var.append({'node': node_id, 'ni': np.nan, 'nj': np.nan, 'nk': np.nan, 'disruption': np.nan,
                                   'in': in_count[node_id], 'out': out_count[node_id]})
                # D[node_id, 0] = id_to_node[node_id]
                # D[node_id, 1] = np.nan
                # D[node_id, 2] = np.nan
                # D[node_id, 3] = np.nan
                # D[node_id, 4] = np.nan
                # D[node_id, 5] = in_count[id_to_node[node_id]]
                # D[node_id, 6] = out_count[id_to_node[node_id]]
            if debug:
                with open("2014_12_core_{}.csv".format(proc), 'a') as output_file:
                    output_file.write('{}\t{}\t{}\n'.format(node_id, in_count[node_id], out_count[node_id]))

            if i % 500000 == 0:
                with open(log_file_path, 'a') as log_file:
                    log_file.write("proc_{} {} {}/{}={}\n".format(proc, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i, total_data, float(i/total_data)))
        except:
            print('Error ocurred in core {} node {}'.format(proc, node_id))
            with open(log_file_path, 'a') as log_file:
                log_file.write('Exception core {} i {} node_id {} in_count[node_id] {} out_count[node_id] {}\n'.format(proc, i, node_id, in_count[node_id], out_count[node_id]))
    print('{} end of core {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), proc))
    with open(log_file_path, 'a') as log_file:
        log_file.write("proc_{} {} end\n".format(proc, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return

def worker_posterior(data_slice, proc, log_file_path):
    # confidences = []
    # _nan = np.nan
    #   https://stackoverflow.com/questions/56683373/parallel-processes-overwriting-progress-bars-tqdm
    #   https://stackoverflow.com/questions/66372005/showing-tqdm-progress-bar-while-using-python-multiprocessing

    #    current = current_process()
    #     for node, ni, nj, nk, disruption, _in, _out in tqdm(data_slice, desc="disruption-{}".format(current.name),
    #                                                         total=len(data_slice), position=current._identity[0] - 1):
    total_data = len(data_slice)
    for i, (node, ni, nj, nk, disruption, _in, _out) in enumerate(data_slice):
        if pd.isna(disruption):
            _posterior_shared_var.append(
                {'node': node, 'ni': ni, 'nj': nj, 'nk': nk, 'disruption': disruption, 'in': _in,
                 'out': _out, 'confidence': np.nan})
        else:
            D = np.random.dirichlet([PRIOR + ni,
                                     PRIOR + nj,
                                     PRIOR + nk], size=10000)
            # pos_i = D[:, 0]
            # pos_j = D[:, 1]
            if disruption <= 0:
                confidence = ((D[:, 0] - D[:, 1]) < 0).mean()
            else:
                confidence = ((D[:, 0] - D[:, 1]) > 0).mean()
            _posterior_shared_var.append({'node': node, 'ni': ni, 'nj': nj, 'nk': nk, 'disruption': disruption, 'in': _in,
                                          'out': _out, 'confidence': confidence})
        if i % 500000 == 0:
            with open(log_file_path, 'a') as log_file:
                log_file.write("proc_{} {} {}/{}={}\n".format(proc, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i, total_data, float(i/total_data)))
    print('{} end of core {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), proc))
    with open(log_file_path, 'a') as log_file:
        log_file.write("proc_{} {} end\n".format(proc, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return

def compute_disruption_multiprocessing_dict_indices(G, year, month, min_in=1, min_out=0, destination_folder='../output', suffix=''):
    global id_to_node, in_count, out_count, node_to_id
    # _disruption_shared_var[:] = []
    # _posterior_shared_var[:] = []

    id_to_node = dict((i, n) for i, n in enumerate(G.nodes))
    node_to_id = dict((v, k) for k, v in id_to_node.items())
    in_count = dict(G.in_degree(G.nodes))
    out_count = dict(G.out_degree(G.nodes))

    start_time = datetime.now()
    # create a dict with in and out links of all nodes in the graph
    print('{} Creating dict with incoming outcoming nodes'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    global node_incoming_outgoing
    node_incoming_outgoing = disruption_utils.create_dict_incoming_outgoing_nodes(G)
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))

    global F
    F = nx.to_scipy_sparse_matrix(G, format='csr')
    # T = nx.to_scipy_sparse_matrix(G, format='csc')
    # D = np.zeros(shape=(F.shape[0], 6))

    # num_proc = multiprocessing.cpu_count() - 10 # Goldorak
    # num_proc = 20 # Goldorak
    num_proc = 70 # Snowden
    range_F = range(F.shape[0])
    n = len(range_F)
    chunk_size = math.floor(n / num_proc)
    _jobs = []

    # print('{} Sequential disruption computation with dict'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # range_F = range(F.shape[0])
    start_time = datetime.now()
    print('{} Multiprocessing disruption computation with dicts'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))

    for p in range(num_proc):
        ini, end = p * chunk_size, chunk_size * (p + 1)

        if p == (num_proc - 1):
            end = n  # if end > n else end
        # print('Total num of rows in matrix {} core {} from {} to {}'.format(n, p, ini, end))
        # data_slice = F[ini: end]
        _process = multiprocessing.Process(target=worker_disruption, args=(ini, end, min_in, min_out, p, ))
        # print('core {} len(_disruption_shared_var) {}'.format(p, len(_disruption_shared_var)))
        _jobs.append(_process)
        _process.start()

    for i in _jobs:
        i.join()
    # print('len(_disruption_shared_var) {}'.format(len(_disruption_shared_var)))
    # print('{}'.format(_disruption_shared_var[0]))
    disrupt = pd.DataFrame(list(_disruption_shared_var))
    # print(disrupt.dtypes)
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))


    cols = ['node', 'ni', 'nj', 'nk', 'disruption', 'in', 'out']
    disruption_values = disrupt[cols].values

    n = len(disruption_values)
    chunk_size = math.floor(n / num_proc)
    _jobs = []
    start_time = datetime.now()
    print('{} Multiprocessing posterior computation with dicts'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))

    for p in range(num_proc):
        ini, end = p * chunk_size, chunk_size * (p + 1)

        if p == (num_proc - 1):
            end = n  # if end > n else end
        # print('Total num of rows in matrix {} core {} from {} to {}'.format(n, p, ini, end))
        data_slice = disruption_values[ini: end]
        _process = multiprocessing.Process(target=worker_posterior, args=(data_slice, ))
        _jobs.append(_process)
        _process.start()

    for i in _jobs:
        i.join()

    disrupt_posterior = pd.DataFrame(list(_posterior_shared_var))
    disrupt_posterior['node'] = disrupt_posterior['node'].astype(np.int64)
    # print(list(disrupt_posterior.columns))
    ## disrupt_posterior.set_index('node', inplace=True)
    # print(disrupt_posterior.dtypes)
    # disrupt_posterior.to_csv('di_mp_dict_{}_{}.csv'.format(year, month), mode='w', sep=',')
    disrupt_posterior.to_csv('{}/di_mp_dict_{}_{}{}.csv'.format(destination_folder, year, month, suffix), mode='w', index=False)
    _disruption_shared_var[:] = []
    _posterior_shared_var[:] = []
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))

def compute_disruption_multiprocessing_dict(G, year, month, num_proc=1, min_in=1, min_out=0, destination_folder='../output', suffix=''):
    global in_count, out_count
    print('number of proc {}'.format(num_proc))

    # id_to_node = dict((i, n) for i, n in enumerate(G.nodes))
    # node_to_id = dict((v, k) for k, v in id_to_node.items())
    in_count = dict(G.in_degree(G.nodes))
    out_count = dict(G.out_degree(G.nodes))
    nodes = list(in_count.keys())

    start_time = datetime.now()
    # create a dict with in and out links of all nodes in the graph
    print('{} Creating dict with incoming outcoming nodes'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    global node_incoming_outgoing
    node_incoming_outgoing = disruption_utils.create_dict_incoming_outgoing_nodes(G)
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))

    # global F
    # F = nx.to_scipy_sparse_matrix(G, format='csr')
    # T = nx.to_scipy_sparse_matrix(G, format='csc')
    # D = np.zeros(shape=(F.shape[0], 6))

    # num_proc = multiprocessing.cpu_count() - 10 # Goldorak
    # num_proc = 20 # Goldorak
    # num_proc = 70 # Snowden
    # range_F = range(F.shape[0])

    # n = len(range_F)
    n = len(nodes)
    chunk_size = math.floor(n / num_proc)
    _jobs = []

    # print('{} Sequential disruption computation with dict'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # range_F = range(F.shape[0])
    start_time = datetime.now()
    print('{} Multiprocessing disruption computation with dicts'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))

    for p in range(num_proc):
        ini, end = p * chunk_size, chunk_size * (p + 1)

        if p == (num_proc - 1):
            end = n  # if end > n else end
        # print('Total num of rows in matrix {} core {} from {} to {}'.format(n, p, ini, end))
        # data_slice = F[ini: end]
        # data_slice = nodes[ini: end]
        _process = multiprocessing.Process(target=worker_disruption, args=(nodes[ini: end], min_in, min_out, p, ))
        # print('core {} len(_disruption_shared_var) {}'.format(p, len(_disruption_shared_var)))
        _jobs.append(_process)
        _process.start()

    for i in _jobs:
        i.join()
    # print('len(_disruption_shared_var) {}'.format(len(_disruption_shared_var)))
    # print('{}'.format(_disruption_shared_var[0]))
    disrupt = pd.DataFrame(list(_disruption_shared_var))
    # print(disrupt.dtypes)

    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))
    start_time = datetime.now()
    # create a dict with in and out links of all nodes in the graph
    print('{} Deleting variables'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    del in_count, out_count, node_incoming_outgoing
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))
    cols = ['node', 'ni', 'nj', 'nk', 'disruption', 'in', 'out']
    disruption_values = disrupt[cols].values

    n = len(disruption_values)
    chunk_size = math.floor(n / num_proc)
    _jobs = []
    start_time = datetime.now()
    print('{} Multiprocessing posterior computation with dicts'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))

    for p in range(num_proc):
        ini, end = p * chunk_size, chunk_size * (p + 1)

        if p == (num_proc - 1):
            end = n  # if end > n else end
        # print('Total num of rows in matrix {} core {} from {} to {}'.format(n, p, ini, end))
        # data_slice = disruption_values[ini: end]
        _process = multiprocessing.Process(target=worker_posterior, args=(disruption_values[ini: end], ))
        _jobs.append(_process)
        _process.start()

    for i in _jobs:
        i.join()

    disrupt_posterior = pd.DataFrame(list(_posterior_shared_var))
    disrupt_posterior['node'] = disrupt_posterior['node'].astype(np.int64)

    disrupt_posterior.to_csv('{}/di_mp_dict_{}_{}{}.csv'.format(destination_folder, year, month, suffix), mode='w', index=False)
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))
    _disruption_shared_var[:] = []
    _posterior_shared_var[:] = []


def compute_disruption_multiprocessing_dict_objects(path_objects, year, month, server, num_proc=1, min_in=1, min_out=0,
                                                    destination_folder='../output', suffix='', debug=False,
                                                    missing=False, missing_nodes=None):
    if missing_nodes is None:
        missing_nodes = []
    debug_label = ''
    if debug:
        debug_label = '_debug'
    log_file_path = '{}/{}_{}_dict_mp{}.log'.format(destination_folder, year, month, debug_label)
    if missing:
        print('{} Computation for missing nodes'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        with open(log_file_path, 'a') as log_file:
            log_file.write("{} Computation for missing nodes\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    global in_count, out_count
    print('id process {}'.format(os.getpid()))
    print('number of proc {}'.format(num_proc))
    with open(log_file_path, 'a') as log_file:
        log_file.write("{} using {} cores\n".format(server, num_proc))
    start_time = datetime.now()
    print('{} Loading in_count, out_count, node_incoming_outgoing'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))

    with open('{}/{}_{}_in_count.pickle'.format(path_objects, year, month), "rb") as file:
        in_count = pickle.load(file)
    print('{} Loaded in_count'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))

    with open('{}/{}_{}_out_count.pickle'.format(path_objects, year, month), "rb") as file:
        out_count = pickle.load(file)
    print('{} Loaded out_count'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    # id_to_node = dict((i, n) for i, n in enumerate(G.nodes))
    # node_to_id = dict((v, k) for k, v in id_to_node.items())

    # in_count = dict(G.in_degree(G.nodes))
    # out_count = dict(G.out_degree(G.nodes))
    if missing:
        nodes = missing_nodes
    else:
        nodes = list(in_count.keys())

    global node_incoming_outgoing
    with open('{}/{}_{}_node_incoming_outgoing.pickle'.format(path_objects, year, month), "rb") as file:
        node_incoming_outgoing = pickle.load(file)

    end_time = datetime.now()
    print('in_count: {}, out_count: {} node_incoming_outgoing: {}'.format(len(in_count), len(out_count),
                                                                          len(node_incoming_outgoing)))
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))
    # start_time = datetime.now()
    # # create a dict with in and out links of all nodes in the graph
    # print('{} Creating dict with incoming outcoming nodes'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    # node_incoming_outgoing = utils.create_dict_incoming_outgoing_nodes(G)
    # end_time = datetime.now()
    # print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))

    # global F
    # F = nx.to_scipy_sparse_matrix(G, format='csr')
    # T = nx.to_scipy_sparse_matrix(G, format='csc')
    # D = np.zeros(shape=(F.shape[0], 6))

    # num_proc = multiprocessing.cpu_count() - 10 # Goldorak
    # num_proc = 20 # Goldorak
    # num_proc = 70 # Snowden
    # range_F = range(F.shape[0])

    # n = len(range_F)
    n = len(nodes)
    chunk_size = math.floor(n / num_proc)
    _jobs = []

    # print('{} Sequential disruption computation with dict'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # range_F = range(F.shape[0])
    start_time = datetime.now()
    print('{} Multiprocessing disruption computation with dicts'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))


    with open(log_file_path, 'a') as log_file:
        log_file.write("{} Starting disruptiveness with dicts in multiprocessing\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    for p in range(num_proc):
        ini, end = p * chunk_size, chunk_size * (p + 1)

        if p == (num_proc - 1):
            end = n  # if end > n else end
        # print('Total num of rows in matrix {} core {} from {} to {}'.format(n, p, ini, end))
        # data_slice = F[ini: end]
        # data_slice = nodes[ini: end]
        _process = multiprocessing.Process(target=worker_disruption, args=(nodes[ini: end], min_in, min_out, p, log_file_path, debug, ))
        # print('core {} len(_disruption_shared_var) {}'.format(p, len(_disruption_shared_var)))
        _jobs.append(_process)
        _process.start()

    for i in _jobs:
        i.join()
    # print('len(_disruption_shared_var) {}'.format(len(_disruption_shared_var)))
    # print('{}'.format(_disruption_shared_var[0]))
    disrupt = pd.DataFrame(list(_disruption_shared_var))
    # print(disrupt.dtypes)
    # print(disrupt.info())
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))
    with open(log_file_path, 'a') as log_file:
        log_file.write("{} end multiprocessing in disruptiveness, lasted {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))

    missing_nodes = set(nodes).difference(set(disrupt['node'].tolist()))
    if len(missing_nodes) > 0:
        save_missing_nodes('{}/missing_nodes_{}_{}.csv'.format(destination_folder, year, month), missing_nodes)
        print('{} there are {} missing nodes'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(missing_nodes)))
        with open(log_file_path, 'a') as log_file:
            log_file.write("{} there are {} missing nodes\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(missing_nodes)))

    start_time = datetime.now()
    # create a dict with in and out links of all nodes in the graph
    print('{} Deleting variables'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    del in_count, out_count, node_incoming_outgoing
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))
    cols = ['node', 'ni', 'nj', 'nk', 'disruption', 'in', 'out']
    disruption_values = disrupt[cols].values
    del disrupt

    n = len(disruption_values)
    chunk_size = math.floor(n / num_proc)
    _jobs = []
    start_time = datetime.now()
    print('{} Multiprocessing posterior computation with dicts'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    with open(log_file_path, 'a') as log_file:
        log_file.write("{} Starting posterior computation with dicts in multiprocessing\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    for p in range(num_proc):
        ini, end = p * chunk_size, chunk_size * (p + 1)

        if p == (num_proc - 1):
            end = n  # if end > n else end
        # print('Total num of rows in matrix {} core {} from {} to {}'.format(n, p, ini, end))
        # data_slice = disruption_values[ini: end]
        _process = multiprocessing.Process(target=worker_posterior, args=(disruption_values[ini: end], p, log_file_path, ))
        _jobs.append(_process)
        _process.start()

    for i in _jobs:
        i.join()

    disrupt_posterior = pd.DataFrame(list(_posterior_shared_var))
    disrupt_posterior['node'] = disrupt_posterior['node'].astype(np.int64)

    path_output_file = '{}/di_mp_dict_{}_{}{}{}.csv'.format(destination_folder, year, month, suffix, debug_label)
    if missing:
        disrupt_posterior.to_csv(path_output_file, mode='a', index=False, header=False)
    else:
        disrupt_posterior.to_csv(path_output_file, mode='w', index=False)
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))
    print('{} is saved'.format(path_output_file))
    with open(log_file_path, 'a') as log_file:
        log_file.write("{} end multiprocessing in posterior, lasted {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))
        log_file.write('{} is saved'.format(path_output_file))
    _disruption_shared_var[:] = []
    _posterior_shared_var[:] = []

def compute_disruption_multiprocessing_dict_objects_debug(path_objects, year, month, num_proc=1, min_in=1, min_out=0, destination_folder='../output', suffix=''):
    global in_count, out_count
    print('id process {}'.format(os.getpid()))
    print('number of proc {}'.format(num_proc))
    month = '0' + str(month) if len(str(month)) == 1 else month

    start_time = datetime.now()
    print('{} Loading in_count, out_count, node_incoming_outgoing'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))

    with open('{}/{}_{}_in_count.pickle'.format(path_objects, year, month), "rb") as file:
        in_count = pickle.load(file)
    with open('{}/{}_{}_out_count.pickle'.format(path_objects, year, month), "rb") as file:
        out_count = pickle.load(file)
    print('{} Loaded in_count, out_count'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    # id_to_node = dict((i, n) for i, n in enumerate(G.nodes))
    # node_to_id = dict((v, k) for k, v in id_to_node.items())

    # in_count = dict(G.in_degree(G.nodes))
    # out_count = dict(G.out_degree(G.nodes))
    nodes = list(in_count.keys())
    print('nodes {}'.format(len(nodes)))
    nodes = nodes[0:1308669]
    print('nodes {}'.format(len(nodes)))

    global node_incoming_outgoing
    with open('{}/{}_{}_node_incoming_outgoing.pickle'.format(path_objects, year, month), "rb") as file:
        node_incoming_outgoing = pickle.load(file)

    print('in_count: {}, out_count: {} node_incoming_outgoing: {}'.format(len(in_count), len(out_count), len(node_incoming_outgoing)))


    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))
    # start_time = datetime.now()
    # # create a dict with in and out links of all nodes in the graph
    # print('{} Creating dict with incoming outcoming nodes'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    # node_incoming_outgoing = utils.create_dict_incoming_outgoing_nodes(G)
    # end_time = datetime.now()
    # print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))

    # global F
    # F = nx.to_scipy_sparse_matrix(G, format='csr')
    # T = nx.to_scipy_sparse_matrix(G, format='csc')
    # D = np.zeros(shape=(F.shape[0], 6))

    # num_proc = multiprocessing.cpu_count() - 10 # Goldorak
    # num_proc = 20 # Goldorak
    # num_proc = 70 # Snowden
    # range_F = range(F.shape[0])

    # n = len(range_F)
    n = len(nodes)
    chunk_size = math.floor(n / num_proc)
    _jobs = []

    # print('{} Sequential disruption computation with dict'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # range_F = range(F.shape[0])
    start_time = datetime.now()
    print('{} Multiprocessing disruption computation with dicts'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    log_file_path = '{}/{}_{}_dict_mp_debug.log'.format(destination_folder, year, month)
    with open(log_file_path, 'a') as log_file:
        log_file.write("{} Starting disruptiveness with dicts in multiprocessing\n".format(datetime.now().strftime("%H:%M:%S")))

    for p in range(num_proc):
        ini, end = p * chunk_size, chunk_size * (p + 1)

        if p == (num_proc - 1):
            end = n  # if end > n else end
        # print('Total num of rows in matrix {} core {} from {} to {}'.format(n, p, ini, end))
        # data_slice = F[ini: end]
        # data_slice = nodes[ini: end]
        _process = multiprocessing.Process(target=worker_disruption, args=(nodes[ini: end], min_in, min_out, p, log_file_path, ))
        # print('core {} len(_disruption_shared_var) {}'.format(p, len(_disruption_shared_var)))
        _jobs.append(_process)
        _process.start()

    for i in _jobs:
        i.join()
    # print('len(_disruption_shared_var) {}'.format(len(_disruption_shared_var)))
    # print('{}'.format(_disruption_shared_var[0]))
    disrupt = pd.DataFrame(list(_disruption_shared_var))
    # print(disrupt.dtypes)

    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))
    with open(log_file_path, 'a') as log_file:
        log_file.write("{} end multiprocessing in disruptiveness\n".format(datetime.now().strftime("%H:%M:%S")))
    start_time = datetime.now()
    # create a dict with in and out links of all nodes in the graph
    print('{} Deleting variables'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    del in_count, out_count, node_incoming_outgoing
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))
    cols = ['node', 'ni', 'nj', 'nk', 'disruption', 'in', 'out']
    disruption_values = disrupt[cols].values

    n = len(disruption_values)
    chunk_size = math.floor(n / num_proc)
    _jobs = []
    start_time = datetime.now()
    print('{} Multiprocessing posterior computation with dicts'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    with open(log_file_path, 'a') as log_file:
        log_file.write("{} Starting posterior computation with dicts in multiprocessing\n".format(datetime.now().strftime("%H:%M:%S")))
    for p in range(num_proc):
        ini, end = p * chunk_size, chunk_size * (p + 1)

        if p == (num_proc - 1):
            end = n  # if end > n else end
        # print('Total num of rows in matrix {} core {} from {} to {}'.format(n, p, ini, end))
        # data_slice = disruption_values[ini: end]
        _process = multiprocessing.Process(target=worker_posterior, args=(disruption_values[ini: end], p, log_file_path, ))
        _jobs.append(_process)
        _process.start()

    for i in _jobs:
        i.join()

    disrupt_posterior = pd.DataFrame(list(_posterior_shared_var))
    disrupt_posterior['node'] = disrupt_posterior['node'].astype(np.int64)

    disrupt_posterior.to_csv('{}/di_mp_dict_{}_{}{}_debug.csv'.format(destination_folder, year, month, suffix), mode='w', index=False)
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))
    with open(log_file_path, 'a') as log_file:
        log_file.write("{} end multiprocessing in posterior\n".format(datetime.now().strftime("%H:%M:%S")))
    _disruption_shared_var[:] = []
    _posterior_shared_var[:] = []

def compute_disruption_sequential_dict(G, year, month, min_in=1, min_out=0, destination_folder='../output', suffix=''):
    id_to_node = dict((i, n) for i, n in enumerate(G.nodes))
    node_to_id = dict((v, k) for k, v in id_to_node.items())
    in_count = dict(G.in_degree(G.nodes))
    out_count = dict(G.out_degree(G.nodes))

    # create a dict with in and out links of all nodes in the graph
    print('{} Creating dict with incoming outcoming nodes'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    node_incoming_outgoing = disruption_utils.create_dict_incoming_outgoing_nodes(G)
    print('{} end'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    F = nx.to_scipy_sparse_matrix(G, format='csr')
    # T = nx.to_scipy_sparse_matrix(G, format='csc')
    D = np.zeros(shape=(F.shape[0], 6))

    print('{} Sequential disruption computation with dict'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    range_F = range(F.shape[0])
    for node_id in tqdm(range_F, desc="disruption computation", total=len(range_F)):
        if in_count[id_to_node[node_id]] >= min_in and \
                out_count[id_to_node[node_id]] >= min_out:
            ni = 0
            nj = 0
            nk = 0

            # outgoing = F[node_id].nonzero()[1]
            outgoing = node_incoming_outgoing[id_to_node[node_id]]['outgoing']
            # incoming = T[:, node_id].nonzero()[0]
            incoming = node_incoming_outgoing[id_to_node[node_id]]['incoming']
            outgoing_set = set(outgoing)

            for other_id in incoming:
                # second_level = F[other_id].nonzero()[1]
                second_level = node_incoming_outgoing[other_id]['outgoing']
                if len(outgoing_set.intersection(second_level)) == 0:
                    ni += 1
                else:
                    nj += 1

            # who mentions my influences
            # who_mentions_my_influences = np.unique(T[:, outgoing].nonzero()[0])
            who_mentions_my_influences = set()
            for outg in outgoing:
                who_mentions_my_influences = who_mentions_my_influences.union(node_incoming_outgoing[outg]['incoming'])
            # who_mentions_my_influences = np.unique(node_incoming_outgoing[id_to_node[outgoing]]['incoming'])
            for other_id in who_mentions_my_influences:
                # do they mention me?! if no, add nk
                # if F[other_id, node_id] == 0 and other_id != node_id:
                if F[node_to_id[other_id], node_id] == 0 and node_to_id[other_id] != node_id:
                    nk += 1

            D[node_id, 0] = ni
            D[node_id, 1] = nj
            D[node_id, 2] = nk
            D[node_id, 3] = (ni - nj) / (ni + nj + nk)
            D[node_id, 4] = in_count[id_to_node[node_id]]
            D[node_id, 5] = out_count[id_to_node[node_id]]
        else:
            D[node_id, 0] = np.nan
            D[node_id, 1] = np.nan
            D[node_id, 2] = np.nan
            D[node_id, 3] = np.nan
            D[node_id, 4] = in_count[id_to_node[node_id]]
            D[node_id, 5] = out_count[id_to_node[node_id]]
    disrupt = pd.DataFrame(D, index=G.nodes, columns=['ni', 'nj', 'nk', 'disruption', 'in', 'out'])

    # disruption = rank_nodes(disrupt)

    print('{} Sequential disruption-posterior computation'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    # print('{} Sequential disruption-posterior computation'.format(datetime.now().strftime("%H:%M:%S")))
    # disruption = rank_nodes_opt(disrupt)
    disruption = rank_nodes(disrupt)
    disrupt['confidence'] = disruption['confidence']
    # print('{} end'.format(datetime.now().strftime("%H:%M:%S")))
    print('{} end'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    disrupt.to_csv('{}/di_seq_dict_{}_{}{}.csv'.format(destination_folder, year, month, suffix), mode='w')
    # print('{} end'.format(datetime.now().strftime("%H:%M:%S")))
    print('{} end'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


def main_process(main_folder_path, year, month, num_proc, destination_folder):
    month = '0' + str(month) if len(str(month)) == 1 else month
    path_nx_graph = '{}/cit_net/{}/cit_net_{}_{}.pickle'.format(main_folder_path, year, year, month)
    if not os.path.exists(path_nx_graph):
        print('{} does not exist in disk'.format(path_nx_graph))
        return
    start_time = datetime.now()
    print('{} {} opening graph'.format(start_time.strftime("%Y-%m-%d %H:%M:%S"), path_nx_graph))
    with open(path_nx_graph, "rb") as file:
        nx_graph = pickle.load(file)
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))

    # disruption_mcn.compute_disruption_sequential(nx_graph, year, month)
    # disrupt = compute_disruption_sequential_dict(nx_graph, year, month)
    compute_disruption_multiprocessing_dict(G=nx_graph, year=year, month=month, num_proc=num_proc,
                                            destination_folder=destination_folder)

def create_monthly_objects(graph_path, output_path, year, month):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    start_time = datetime.now()
    print('{} {} opening graph'.format(start_time.strftime("%Y-%m-%d %H:%M:%S"), graph_path))
    with open(graph_path, "rb") as file:
        nx_graph = pickle.load(file)
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))

    # create a dict with in and out links of all nodes in the graph
    start_time = datetime.now()
    print('{} Creating dict with incoming outcoming nodes'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    node_incoming_outgoing = disruption_utils.create_dict_incoming_outgoing_nodes(nx_graph)
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))

    start_time = datetime.now()
    print('{} Creating in_count, out_count'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    in_count = dict(nx_graph.in_degree(nx_graph.nodes))
    out_count = dict(nx_graph.out_degree(nx_graph.nodes))
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))

    start_time = datetime.now()
    print('{} Creating pickle files in disk'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    with open('{}/{}_{}_{}.pickle'.format(output_path, year, month, 'node_incoming_outgoing'), 'wb') as file:
        pickle.dump(node_incoming_outgoing, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open('{}/{}_{}_{}.pickle'.format(output_path, year, month, 'in_count'), 'wb') as file:
        pickle.dump(in_count, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open('{}/{}_{}_{}.pickle'.format(output_path, year, month, 'out_count'), 'wb') as file:
        pickle.dump(out_count, file, protocol=pickle.HIGHEST_PROTOCOL)
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))
    del nx_graph, node_incoming_outgoing, in_count, out_count


def load_missing_nodes(path_missing_nodes):
    missing_nodes = []
    with open(path_missing_nodes) as missing_nodes_file:
        for node in missing_nodes_file:
            missing_nodes.append(int(node.strip()))
    return missing_nodes


def save_missing_nodes(path_file, missing_nodes):
    with open(path_file, 'w') as missing_nodes_file:
        for node in missing_nodes:
            missing_nodes_file.write('{}\n'.format(node))


def compute_disruptiveness_mp_dict_objects_month(objects_folder, year, month, server, num_proc, destination_folder, debug=False):
    compute_disruption_multiprocessing_dict_objects(path_objects=objects_folder, year=year, month=month,
                                                    server=server, num_proc=num_proc,
                                                    destination_folder=destination_folder, debug=False)
    if os.path.exists('{}/missing_nodes_{}_{}.csv'.format(destination_folder, year, month)):
        missing_nodes = load_missing_nodes('{}/missing_nodes_{}_{}.csv'.format(destination_folder, year, month))
        if len(missing_nodes) > 0:
            compute_disruption_multiprocessing_dict_objects(path_objects=objects_folder, year=year, month=month,
                                                            num_proc=num_proc, server=server,
                                                            destination_folder=destination_folder, debug=False,
                                                            missing=True, missing_nodes=missing_nodes)

def main():
    # # running in Goldorak: tm_ref2
    # year_s, year_e, month_s, month_e = 2014, 2014, 3, 12
    # # year, month = 1961, 1
    # for year in range(year_s, year_e + 1):
    #     for month in range(month_s, month_e + 1):
    #         main_folder_path = '/data/user/copara/dataset/PubMed/2022'
    #         # main_folder_path = '/home/cineca/Documents/dataset/PubMed/2022'#'/home/cineca/PycharmProjects/graphs/output'
    #         main_process(year=year, month=month, main_folder_path=main_folder_path)

    # running in Snowden: sc_cn
    # main_folder_path = '/data/user/copara/dataset/PubMed/2022'

    # main_folder_path = '/home/cineca/Documents/dataset/PubMed/2022'#'/home/cineca/PycharmProjects/graphs/output'

    # year_s, year_e, month_s, month_e = 2014, 2014, 1, 2 # Goldorak, still doing at 18h45
    # year_s, year_e, month_s, month_e = 2015, 2015, 8, 8  # Goldorak, to run
    # destination_folder = '/data/user/copara/dataset/PubMed/2022/disruption_obj'
    # # destination_folder = '/home/copara/dataset/PubMed/2022/disruption_obj'
    # print('Creating monthly objects for disruption computation')
    # for year in range(year_s, year_e + 1):
    #     destination_folder = '{}/{}'.format(destination_folder, year)
    #     for month in range(month_s, month_e + 1):
    #         month = '0' + str(month) if len(str(month)) == 1 else month
    #         path_nx_graph = '{}/cit_net/{}/cit_net_{}_{}.pickle'.format(main_folder_path, year, year, month)
    #         if not os.path.exists(path_nx_graph):
    #             print('{} does not exist in disk'.format(path_nx_graph))
    #             return
    #         create_monthly_objects(graph_path=path_nx_graph, output_path=destination_folder, year=year, month=month)

    # year_s, year_e, month_s, month_e = 2015, 2015, 5, 6
    # main_folder_path = '/home/copara/dataset/PubMed/2022'
    # destination_folder = '/home/copara/dataset/PubMed/2022/disruption_obj'
    # print('Creating monthly objects for disruption computation')
    # for year in range(year_s, year_e + 1):
    #     destination_folder = '{}/{}'.format(destination_folder, year)
    #     for month in range(month_s, month_e + 1):
    #         month = '0' + str(month) if len(str(month)) == 1 else month
    #         path_nx_graph = '{}/cit_net/{}/cit_net_{}_{}.pickle'.format(main_folder_path, year, year, month)
    #         if not os.path.exists(path_nx_graph):
    #             print('{} does not exist in disk'.format(path_nx_graph))
    #             return
    #         create_monthly_objects(graph_path=path_nx_graph, output_path=destination_folder, year=year, month=month)
    # year_s, year_e, month_s, month_e = 2015, 2015, 9, 12
    # destination_folder = '/home/copara/dataset/PubMed/2022/disruption_obj'
    # print('Creating monthly objects for disruption computation')
    # for year in range(year_s, year_e + 1):
    #     destination_folder = '{}/{}'.format(destination_folder, year)
    #     for month in range(month_s, month_e + 1):
    #         month = '0' + str(month) if len(str(month)) == 1 else month
    #         path_nx_graph = '{}/cit_net/{}/cit_net_{}_{}.pickle'.format(main_folder_path, year, year, month)
    #         if not os.path.exists(path_nx_graph):
    #             print('{} does not exist in disk'.format(path_nx_graph))
    #             return
    #         create_monthly_objects(graph_path=path_nx_graph, output_path=destination_folder, year=year, month=month)


    # Goldorak tm_ref2
    # num_proc = 10 # Goldorak
    #
    # year_s, year_e, month_s, month_e = 2015, 2015, 7, 8
    # # destination_folder = '/home/copara/code/graphs/output'
    # destination_folder = '/data/user/copara/code/projects/graphs/output'
    # # objects_folder = '/home/copara/dataset/PubMed/2022/disruption_obj'
    # objects_folder = '/data/user/copara/dataset/PubMed/2022/disruption_obj'
    # #
    # for year in range(year_s, year_e + 1):
    #     objects_folder = '{}/{}'.format(objects_folder, year)
    #     for month in range(month_s, month_e + 1):
    #         print('{} {}-{}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), year, month))
    #         # main_process(year=year, month=month, main_folder_path=main_folder_path, num_proc=num_proc,
    #         #              destination_folder=destination_folder)
    #         compute_disruption_multiprocessing_dict_objects(path_objects=objects_folder, year=year, month=month,
    #                                                         num_proc=num_proc, destination_folder=destination_folder)
    #

    # # objects_folder = '/home/copara/dataset/PubMed/2022/disruption_obj'

    # Goldorak tm_ref2
    num_proc = 5
    year_s, year_e, month_s, month_e = 2020, 2020, 3, 12
    # destination_folder = '/home/copara/code/graphs/output'
    destination_folder = '/data/user/copara/code/projects/graphs/output'
    objects_folder = '/data/user/copara/dataset/PubMed/2022/disruption_obj'
    server = 'Goldorak'
    #
    # Snowden sc_ref2
    # num_proc = 2 # Snowden
    # year_s, year_e, month_s, month_e = 2020, 2020, 1, 2
    # destination_folder = '/home/copara/code/graphs/output'
    # objects_folder = '/home/copara/dataset/PubMed/2022/disruption_obj'
    # server = 'Snowden'
    #
    # Baobab
    # num_proc = 8
    # year_s, year_e, month_s, month_e = 1951, 1951, 4, 8
    # destination_folder = '/home/users/c/coparaz9/dataset/PubMed/2022/metrics'
    # objects_folder = '/home/users/c/coparaz9/dataset/PubMed/2022/disruption_obj'
    # server = 'Baobab'

    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    destination_folder = '{}/disruption'.format(destination_folder)
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    for year in range(year_s, year_e + 1):
        objects_folder = '{}/{}'.format(objects_folder, year)
        destination_folder_year = '{}/{}'.format(destination_folder, year)
        if not os.path.exists(destination_folder_year):
            os.mkdir(destination_folder_year)
        for month in range(month_s, month_e + 1):
            month = '0' + str(month) if len(str(month)) == 1 else month
            print('{} {} {}-{}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), server, year, month))
            # main_process(year=year, month=month, main_folder_path=main_folder_path, num_proc=num_proc,
            #              destination_folder=destination_folder)
            compute_disruption_multiprocessing_dict_objects(path_objects=objects_folder, year=year, month=month,
                                                            server=server, num_proc=num_proc,
                                                            destination_folder=destination_folder_year, debug=False)
            if os.path.exists('{}/missing_nodes_{}_{}.csv'.format(destination_folder_year, year, month)):
                missing_nodes = load_missing_nodes('{}/missing_nodes_{}_{}.csv'.format(destination_folder_year, year, month))
                if len(missing_nodes) > 0:
                    compute_disruption_multiprocessing_dict_objects(path_objects=objects_folder, year=year, month=month,
                                                                    num_proc=num_proc, server=server,
                                                                    destination_folder=destination_folder_year, debug=False,
                                                                    missing=True, missing_nodes=missing_nodes)

    # year_s, year_e, month_s, month_e = 2015, 2015, 1, 6
    # destination_folder = '/home/copara/code/graphs/output'
    # objects_folder = '/home/copara/dataset/PubMed/2022/disruption_obj'
    # for year in range(year_s, year_e + 1):
    #     objects_folder = '{}/{}'.format(objects_folder, year)
    #     for month in range(month_s, month_e + 1):
    #         print('{} {}-{}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), year, month))
    #         # main_process(year=year, month=month, main_folder_path=main_folder_path, num_proc=num_proc,
    #         #              destination_folder=destination_folder)
    #         compute_disruption_multiprocessing_dict_objects(path_objects=objects_folder, year=year, month=month,
    #                                                         num_proc=num_proc, destination_folder=destination_folder)

    # year_s, year_e, month_s, month_e = 1961, 1961, 1, 1
    # # year, month = 1961, 1
    # for year in range(year_s, year_e + 1):
    #     for month in range(month_s, month_e + 1):
    #         # main_folder_path = '/data/user/copara/dataset/PubMed/2022' ffff
    #         # main_folder_path = '/home/cineca/Documents/dataset/PubMed/2022'#'/home/cineca/PycharmProjects/graphs/output'
    #         main_folder_path = '/home/copara/dataset/PubMed/2022'
    #         main_process(year=year, month=month, main_folder_path=main_folder_path)

if __name__ == '__main__':
    main()