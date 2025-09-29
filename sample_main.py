# from metrics_utils import sample_nodes as _sample_nodes
import pickle
import os
from datetime import datetime

import random
import math
random.seed(10)

# import centralities
from centralities.graph_metrics import compute_selected_centralities

from graph_utils import create_monthly_objects
from entropy.entropy_hierarchy import compute_entropy_hierarchy, load_tree_numbers
from CU.cu_hierarchy import compute_cu

def _sample_nodes(nodes: list, portion=0.1):
    k_length = math.ceil(portion*len(nodes))
    return random.sample(nodes, k_length)


def create_sample_graph(main_folder_path, year, month, destination_folder):
    path_nx_graph = '{}/cit_net/{}/cit_net_{}_{}.pickle'.format(main_folder_path, year, year, month)
    if not os.path.exists(path_nx_graph):
        print('{} does not exist in disk'.format(path_nx_graph))
        return

    start_time = datetime.now()
    print('{} {} opening graph'.format(start_time.strftime("%Y-%m-%d %H:%M:%S"), path_nx_graph))
    # load citation network from pickle file
    with open(path_nx_graph, "rb") as file:
        nx_graph = pickle.load(file)
    end_time = datetime.now()
    print('{} end, lasted'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), end_time-start_time))

    start_time = datetime.now()
    print('{} Generating sample'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    sample_nodes = _sample_nodes(nodes=list(nx_graph), portion=0.1)
    sampled_graph = nx_graph.subgraph(sample_nodes).copy()
    print('{}-{} num nodes {}, num edges {}'.format(year, month, len(sampled_graph), len(sampled_graph.edges())))

    # save sampled citation network
    if not os.path.exists('{}'.format(destination_folder)):
        os.mkdir('{}'.format(destination_folder))
    if not os.path.exists('{}/{}'.format(destination_folder, year)):
        os.mkdir('{}/{}'.format(destination_folder, year))
    sample_filename = '{}/{}/cit_net_sample_{}_{}.pickle'.format(destination_folder, year, year, month)
    with open(sample_filename, 'wb') as file:
        pickle.dump(sampled_graph, file, protocol=pickle.HIGHEST_PROTOCOL)
    end_time = datetime.now()
    print('{} saving {}'.format(end_time.strftime("%Y-%m-%d %H:%M:%S"), sample_filename))
    print('sample generation lasted {}'.format(end_time-start_time))

    # nx_graph = sampled_graph
    # suffix_disruption_file = '_sample'
    # return sampled_graph


def test_open(path_nx_graph):
    print('{} {} opening graph'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), path_nx_graph))
    # load citation network from pickle file
    with open(path_nx_graph, "rb") as file:
        nx_graph = pickle.load(file)
    print('{} end'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    print('{} num nodes {}, num edges {}'.format(path_nx_graph, len(nx_graph), len(nx_graph.edges())))
    # print('!!!')





def create_folder_if_does_not_exist(path_folder):
    if not os.path.exists(path_folder):
        os.mkdir(path_folder)




def main_create_sampled_graphs():
    # general variables
    year_s, year_e, month_s, month_e = 2019, 2021, 1, 12
    server = 'goldorak'  # laptop, goldorak, snowden
    label_sample = '_sample'
    if server == 'laptop':
        main_folder_path = '/home/cineca/Documents/dataset/PubMed/2022'
    elif server == 'goldorak':
        main_folder_path = '/data/user/copara/dataset/PubMed/2022'
    else:
        main_folder_path = ''
    print('Creating sample graphs from {}-{} to {}-{}'.format(year_s, month_s, year_e, month_e))
    destination_folder_sample_graph = '{}/cit_net{}'.format(main_folder_path, label_sample)
    for year in range(year_s, year_e + 1):
        for month in range(month_s, month_e + 1):
            month = '0' + str(month) if len(str(month)) == 1 else month
            print('--------------{}-{}--------------'.format(year, month))
            sample_filename = '{}/{}/cit_net_sample_{}_{}.pickle'.format(destination_folder_sample_graph, year, year, month)
            if not os.path.exists(sample_filename):
                create_sample_graph(main_folder_path, year, month, destination_folder_sample_graph)
            else:
                print("Sample already generated")


def main_compute_metrics(year_s, year_e, month_s, month_e, is_sample, server='laptop', num_proc=8):
    # general variables
    # year_s, year_e, month_s, month_e = 2021, 2021, 1, 6
    # is_sample = True
    # server = 'goldorak'  # laptop, goldorak, snowden
    # num_proc = 8  # number of cores to use for computation of disruptiveness

    label_sample = '_sample' if is_sample else ''
    if server == 'laptop':
        main_folder_path = '/home/cineca/Documents/dataset/PubMed/2022'
        pmid_meshterms_file = '/home/cineca/PycharmProjects/graphs/objects/pmid_meshterms.pickle' #'/data/user/copara/code/projects/graphs/objects/pmid_meshterms.pickle'
        tree_numbers_file_path = '/home/cineca/Documents/dataset/MeSH_descriptor_data/md_treenumbers_2022.txt' # '/data/user/copara/dataset/MeSH/md_treenumbers_2022.txt'
        # CHANGE according to metric that is computed
        label_logfile = 'sample' if is_sample else 'full'
        logfile_path = '/home/cineca/PycharmProjects/graphs/output/metrics_{}_graph_output_L_{}-{}_{}-{}.txt'.\
            format(label_logfile, year_s, year_e, month_s, month_e)
    elif server == 'goldorak':
        main_folder_path = '/data/user/copara/dataset/PubMed/2022'
        pmid_meshterms_file = '/data/user/copara/code/projects/graphs/objects/pmid_meshterms.pickle'
        tree_numbers_file_path = '/data/user/copara/dataset/MeSH/md_treenumbers_2022.txt'
        # CHANGE according to metric that is computed
        label_logfile = 'sample' if is_sample else 'full'
        logfile_path = '/data/user/copara/code/projects/graphs/output/metrics_{}_graph_output_G_{}-{}_{}-{}.txt'.\
            format(label_logfile, year_s, year_e, month_s, month_e)
    else:
        main_folder_path = ''
        pmid_meshterms_file = ''
        tree_numbers_file_path = ''
        logfile_path = ''

    destination_folder_sample_graph = '{}/cit_net{}'.format(main_folder_path, label_sample)
    metrics_path = '{}/metrics{}'.format(main_folder_path, label_sample)

    # disruption variables
    dest_folder_disruption_obj = '{}/disruption_obj{}'.format(main_folder_path, label_sample) #'/data/user/copara/dataset/PubMed/2022/disruption_obj{}'
    dest_folder_disruption = '{}/disruption'.format(metrics_path)
    # objects_folder = dest_folder_disruption_obj

    create_folder_if_does_not_exist(dest_folder_disruption_obj)
    create_folder_if_does_not_exist(metrics_path)
    # if not os.path.exists(dest_folder_disruption_obj):
    #     os.mkdir(dest_folder_disruption_obj)
    create_folder_if_does_not_exist(dest_folder_disruption)

    # entropy variables
    # main_folder_path = '/data/user/copara/dataset/PubMed/2022'
    output_folder_path_entropy = '{}/entropy'.format(metrics_path) #'/data/user/copara/code/projects/graphs/output'
    print('See log file in {}'.format(logfile_path))
    with open(logfile_path, 'w') as log_file:
        start_time = datetime.now()
        log_file.write('{} Loading pmid_meshterms pickle'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
        print('{} Loading pmid_meshterms pickle'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
        with open(pmid_meshterms_file, "rb") as file:
            pmid_meshterms = pickle.load(file)
        end_time = datetime.now()
        log_file.write('{} end, lasted {}'.format(end_time.strftime("%H:%M:%S"), end_time-start_time))
        print('{} end, lasted {}'.format(end_time.strftime("%H:%M:%S"), end_time-start_time))
        tree_numbers = load_tree_numbers(tree_numbers_file_path=tree_numbers_file_path)
        create_folder_if_does_not_exist(output_folder_path_entropy)

        # CU variables
        main_cn_folder_path = '{}/cit_net{}'.format(main_folder_path, label_sample)
        output_folder_path_cu = '{}/CU'.format(metrics_path)
        prefix_input_filename, prefix_output_filename = 'cit_net{}'.format(label_sample), 'hierarchy_cu{}'.format(label_sample)
        create_folder_if_does_not_exist(output_folder_path_cu)

        for year in range(year_s, year_e + 1):
            # create year folder if it does not exist
            dest_folder_disruption_obj_year = '{}/{}'.format(dest_folder_disruption_obj, year)
            create_folder_if_does_not_exist(dest_folder_disruption_obj_year)
            dest_folder_disruption_year = '{}/{}'.format(dest_folder_disruption, year)
            create_folder_if_does_not_exist(dest_folder_disruption_year)

            output_folder_path_entropy_year = '{}/{}'.format(output_folder_path_entropy, year)
            create_folder_if_does_not_exist(output_folder_path_entropy_year)

            output_folder_path_cu_year = '{}/{}'.format(output_folder_path_cu, year)
            create_folder_if_does_not_exist(output_folder_path_cu_year)

            for month in range(month_s, month_e + 1):
                month = '0' + str(month) if len(str(month)) == 1 else month
                log_file.write('--------------{}-{}--------------'.format(year, month))
                print('--------------{}-{}--------------'.format(year, month))

                sample_filename = '{}/{}/cit_net_sample_{}_{}.pickle'.format(destination_folder_sample_graph, year, year, month)
                if is_sample and not os.path.exists(sample_filename):
                    create_sample_graph(main_folder_path, year, month, destination_folder_sample_graph)

                # path_nx_graph = '{}/{}/cit_net{}_{}_{}.pickle'.format(main_cn_folder_path, year, label_sample, year, month)
                # print('::::::::Pagerank::::::::')
                # compute_selected_centralities(path_nx_graph=path_nx_graph, year=year, month=month, output_path=metrics_path)
                #
                # print('::::::::Disruption::::::::')
                # in_count_path = '{}/{}_{}_in_count.pickle'.format(dest_folder_disruption_obj_year, year, month)
                # incoming_outgoing_path = '{}/{}_{}_in_count.pickle'.format(dest_folder_disruption_obj_year, year, month)
                # out_count_path = '{}/{}_{}_in_count.pickle'.format(dest_folder_disruption_obj_year, year, month)
                # if not os.path.exists(in_count_path) and not os.path.exists(incoming_outgoing_path) and not os.path.exists(out_count_path):
                #     create_monthly_objects(graph_path=path_nx_graph, year=year, month=month, output_path=dest_folder_disruption_obj_year)
                # print('in_count file {}'.format(in_count_path))
                # print('out_count file {}'.format(out_count_path))
                # print('incoming_outgoing file {}'.format(incoming_outgoing_path))
                # from disruption.di_mcn_dict_mp import compute_disruptiveness_mp_dict_objects_month
                # compute_disruptiveness_mp_dict_objects_month(objects_folder=dest_folder_disruption_obj_year, year=year, month=month,
                #                                              server=server, num_proc=num_proc, destination_folder=dest_folder_disruption)
                #
                print('::::::::Entropy::::::::')
                compute_entropy_hierarchy(main_folder_path=main_cn_folder_path, year=year, month=month,
                                          pmid_meshterms=pmid_meshterms,
                                          tree_numbers=tree_numbers, output_folder=output_folder_path_entropy_year, sample=is_sample)

                log_file.write('::::::::CU::::::::')
                print('::::::::CU::::::::')
                output_log_cu = compute_cu(main_cn_folder_path, year, month, pmid_meshterms, tree_numbers,
                                           output_folder_path_cu_year, prefix_input_filename=prefix_input_filename,
                                           prefix_output_filename=prefix_output_filename)
                log_file.write(output_log_cu)
                print(output_log_cu)

def main_compute_metrics_mesh_versions(year_s, year_e, month_s, month_e, is_sample, server='', num_proc=8):
    # general variables
    # year_s, year_e, month_s, month_e = 2021, 2021, 1, 6
    # is_sample = True
    # server = 'goldorak'  # laptop, goldorak, snowden
    # num_proc = 8  # number of cores to use for computation of disruptiveness

    label_sample = '_sample' if is_sample else ''
    if server == 'laptop':
        main_folder_path = './data'
        descriptor_path = './data'
        main_folder_path_pm2022 = './data'
        destination_metrics = './output'
    else:
        print('Choose a machine to run code')


    # CHANGE according to metric that is computed
    label_logfile = 'sample' if is_sample else 'full'
    logfile_path = './output/metrics_{}_graph_output_L_{}-{}_{}-{}.txt'.\
        format(label_logfile, year_s, year_e, month_s, month_e)
    # elif server == 'goldorak':
    #     main_folder_path = '/data/user/copara/dataset/PubMed/2022'
    #     pmid_meshterms_file = '/data/user/copara/code/projects/graphs/objects/pmid_meshterms.pickle'
    #     tree_numbers_file_path = '/data/user/copara/dataset/MeSH/md_treenumbers_2022.txt'
    #     # CHANGE according to metric that is computed
    #     label_logfile = 'sample' if is_sample else 'full'
    #     logfile_path = '/data/user/copara/code/projects/graphs/output/metrics_{}_graph_output_G_{}-{}_{}-{}.txt'.\
    #         format(label_logfile, year_s, year_e, month_s, month_e)
    # else:
    #     # main_folder_path = ''
    #     # pmid_meshterms_file = ''
    #     # tree_numbers_file_path = ''
    #     # logfile_path = ''
    #     return

    destination_folder_sample_graph = '{}/cit_net{}'.format(main_folder_path_pm2022, label_sample)
    metrics_path = '{}/metrics{}'.format(destination_metrics, label_sample)

    # disruption variables
    dest_folder_disruption_obj = '{}/disruption_obj{}'.format(main_folder_path, label_sample) #'/data/user/copara/dataset/PubMed/2022/disruption_obj{}'
    dest_folder_disruption = '{}/disruption'.format(metrics_path)
    # objects_folder = dest_folder_disruption_obj

    create_folder_if_does_not_exist(dest_folder_disruption_obj)
    create_folder_if_does_not_exist(metrics_path)
    # if not os.path.exists(dest_folder_disruption_obj):
    #     os.mkdir(dest_folder_disruption_obj)
    create_folder_if_does_not_exist(dest_folder_disruption)

    # entropy variables
    output_folder_path_entropy = '{}/entropy'.format(metrics_path) #'/data/user/copara/code/projects/graphs/output'
    print('See log file in {}'.format(logfile_path))
    with open(logfile_path, 'w') as log_file:


        create_folder_if_does_not_exist(output_folder_path_entropy)

        # CU variables
        main_cn_folder_path = '{}/cit_net{}'.format(main_folder_path_pm2022, label_sample)
        output_folder_path_cu = '{}/CU'.format(metrics_path)
        prefix_input_filename, prefix_output_filename = 'cit_net{}'.format(label_sample), 'hierarchy_cu{}'.format(label_sample)
        create_folder_if_does_not_exist(output_folder_path_cu)

        for year in range(year_s, year_e + 1):
            start_time = datetime.now()
            pmid_meshterms_year_file = '{}/objects/{}_pmid_meshterms.pickle'.format(main_folder_path, year)
            tree_numbers_year_file_path = '{}/objects/md_treenumbers_{}.txt'.format(descriptor_path, year)
            log_file.write('{} Loading pmid_meshterms pickle'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
            print('{} Loading pmid_meshterms pickle'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
            with open(pmid_meshterms_year_file, "rb") as file:
                pmid_meshterms_year = pickle.load(file)
            end_time = datetime.now()
            log_file.write('{} end, lasted {}'.format(end_time.strftime("%H:%M:%S"), end_time - start_time))
            print('{} end, lasted {}'.format(end_time.strftime("%H:%M:%S"), end_time - start_time))
            tree_numbers_year = load_tree_numbers(tree_numbers_file_path=tree_numbers_year_file_path)


            # create year folder if it does not exist
            dest_folder_disruption_obj_year = '{}/{}'.format(dest_folder_disruption_obj, year)
            create_folder_if_does_not_exist(dest_folder_disruption_obj_year)
            dest_folder_disruption_year = '{}/{}'.format(dest_folder_disruption, year)
            create_folder_if_does_not_exist(dest_folder_disruption_year)

            output_folder_path_entropy_year = '{}/{}'.format(output_folder_path_entropy, year)
            create_folder_if_does_not_exist(output_folder_path_entropy_year)

            output_folder_path_cu_year = '{}/{}'.format(output_folder_path_cu, year)
            create_folder_if_does_not_exist(output_folder_path_cu_year)

            for month in range(month_s, month_e + 1):
                month = '0' + str(month) if len(str(month)) == 1 else month
                log_file.write('--------------{}-{}--------------'.format(year, month))
                print('--------------{}-{}--------------'.format(year, month))

                sample_filename = '{}/{}/cit_net_sample_{}_{}.pickle'.format(destination_folder_sample_graph, year, year, month)
                if is_sample and not os.path.exists(sample_filename):
                    create_sample_graph(main_folder_path, year, month, destination_folder_sample_graph)

                path_nx_graph = '{}/{}/cit_net{}_{}_{}.pickle'.format(main_cn_folder_path, year, label_sample, year, month)
                print('::::::::Pagerank::::::::')
                compute_selected_centralities(path_nx_graph=path_nx_graph, year=year, month=month, output_path=metrics_path)

                print('::::::::Disruption::::::::')
                in_count_path = '{}/{}_{}_in_count.pickle'.format(dest_folder_disruption_obj_year, year, month)
                incoming_outgoing_path = '{}/{}_{}_in_count.pickle'.format(dest_folder_disruption_obj_year, year, month)
                out_count_path = '{}/{}_{}_in_count.pickle'.format(dest_folder_disruption_obj_year, year, month)
                if not os.path.exists(in_count_path) and not os.path.exists(incoming_outgoing_path) and not os.path.exists(out_count_path):
                    create_monthly_objects(graph_path=path_nx_graph, year=year, month=month, output_path=dest_folder_disruption_obj_year)
                print('in_count file {}'.format(in_count_path))
                print('out_count file {}'.format(out_count_path))
                print('incoming_outgoing file {}'.format(incoming_outgoing_path))
                from disruption.di_mcn_dict_mp import compute_disruptiveness_mp_dict_objects_month
                compute_disruptiveness_mp_dict_objects_month(objects_folder=dest_folder_disruption_obj_year, year=year, month=month,
                                                             server=server, num_proc=num_proc, destination_folder=dest_folder_disruption_year)


                print('::::::::Entropy::::::::')
                compute_entropy_hierarchy(main_folder_path=main_cn_folder_path, year=year, month=month,
                                          pmid_meshterms=pmid_meshterms_year,
                                          tree_numbers=tree_numbers_year, output_folder=output_folder_path_entropy_year, sample=is_sample)

                log_file.write('::::::::CU::::::::')
                print('::::::::CU::::::::')
                output_log_cu = compute_cu(main_cn_folder_path, year, month, pmid_meshterms_year, tree_numbers_year,
                                           output_folder_path_cu_year, prefix_input_filename=prefix_input_filename,
                                           prefix_output_filename=prefix_output_filename)
                log_file.write(output_log_cu)
                print(output_log_cu)


if __name__ == '__main__':
    # # year_s, year_e, month_s, month_e = 2021, 2021, 1, 6
    # # is_sample = True
    # # main_compute_metrics(year_s=2021, year_e=2021, month_s=1, month_e=5, is_sample=True) # finished
    # main_compute_metrics(year_s=2018, year_e=2018, month_s=2, month_e=12, is_sample=False)  # writing in output/metrics_full_graph_output_G_2018-2020_CU.txt # previous->output/metrics_full_graph_output_G_2015-2020_CU.txt
    # main_compute_metrics(year_s=2019, year_e=2020, month_s=1, month_e=12, is_sample=False)  # writing in output/metrics_full_graph_output_G_2018-2020_CU.txt # previous->output/metrics_full_graph_output_G_2015-2020_CU.txt
    # main_compute_metrics(year_s=2021, year_e=2021, month_s=1, month_e=11, is_sample=False)  # writing in output/metrics_full_graph_output_G_2021_01-11_CU.txt # previous->output/metrics_full_graph_output_G_2015-2020_CU.txt
    # # main_create_sampled_graphs()
    # # get_summary_sample()

    year_s, year_e, month_s, month_e = 2014, 2014, 1, 2
    main_compute_metrics_mesh_versions(year_s, year_e, month_s, month_e, is_sample=True, server='laptop')
