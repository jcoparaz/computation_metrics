from datetime import datetime
from tqdm import tqdm
import os
import pickle
import subprocess
import pandas as pd

'''
Function to extract the tree numbers from the file bin of MeSH
e.g., /home/cineca/Documents/dataset/MeSH_descriptor_data/d2021.bin
output: file with lines in the form mesh_descriptor tree_number1 tree_number2 ...
e.g., D000016	C16.131.080 C26.733.031 G01.750.748.500.031 N06.850.460.350.850.500.031 N06.850.810.300.360.031
'''
def get_tree_numbers_mesh(input_file_path, output_file_path):
    with open(input_file_path) as descriptor_file:
        mesh_content = descriptor_file.read().split('*NEWRECORD')
    # print(mesh_content)
    tree_numbers = {}
    for line in mesh_content:
        line = line.strip()
        if line == '':
            continue
        record = line.split('\n')
        tree_str, unq_id = '', ''
        for r in record:
            r = r.split('=')
            if r[0].strip() == 'MN':
                tree_str += r[1].strip() + ' '
            elif r[0].strip() == 'UI':
                unq_id = r[1]
        if unq_id not in tree_numbers:
            tree_numbers[unq_id] = tree_str
        # print(line)
    with open(output_file_path, 'wt') as tree_numbers_file:
        for ui, tn in tree_numbers.items():
            tree_numbers_file.write('{}\t{}\n'.format(ui.strip(), tn.strip()))

'''
Function to extract the mesh headings from the file bin of MeSH
e.g., /home/cineca/Documents/dataset/MeSH_descriptor_data/d2021.bin
output: file with lines in the form mesh_descriptor tree_number1 tree_number2 ...
e.g., D017624	WAGR Syndrome
'''
def get_mesh_heading_bin_mesh(input_file_path, output_file_path):
    with open(input_file_path) as descriptor_file:
        mesh_content = descriptor_file.read().split('*NEWRECORD')
    # print(mesh_content)
    mesh_heading = {}
    for line in mesh_content:
        line = line.strip()
        if line == '':
            continue
        record = line.split('\n')
        mesh_head_str, unq_id = '', ''
        for r in record:
            r = r.split('=')
            if r[0].strip() == 'MH':
                mesh_head_str = r[1].strip()
            elif r[0].strip() == 'UI':
                unq_id = r[1]
        if unq_id not in mesh_heading:
            mesh_heading[unq_id] = mesh_head_str
        # print(line)
    with open(output_file_path, 'wt') as mesh_headings_file:
        for ui, tn in mesh_heading.items():
            mesh_headings_file.write('{}\t{}\n'.format(ui.strip(), tn.strip()))


def wccount(filename):
    out = subprocess.Popen(['wc', '-l', filename],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT
                         ).communicate()[0]
    return int(out.partition(b' ')[0])


def create_pmid_meshterms(main_path, output_path, year_s=1781, year_e=2021, month_s=1, month_e=12, repeated_path=''):
    pmid_meshterms = {}
    repeated_output = ''
    print("{} loading pmid_date dict".format(datetime.now().strftime("%H:%M:%S")))
    range_years = range(year_s, year_e + 1)
    range_months = range(month_s, month_e + 1)
    for y in tqdm(range_years, desc=" years", total=len(range_years)):
        for m in tqdm(range_months, desc="{} months".format(y), total=len(range_months)):
            if not os.path.exists('{}{}/{}/'.format(main_path, y, m)):
                continue
            unique_record_file_path = '{}{}/{}/pmids.csv'.format(main_path, y, m)
            total_lines = wccount(unique_record_file_path)
            with open(unique_record_file_path) as pmids_file:
                for line in tqdm(pmids_file, desc=" PMIDs", total=total_lines):
                    # 1951-04-01	14832662	D001794 D001795 D005260 D006062 D006973 D011225 D011247 D018805 D014115 	The humoral origin of hypertension in toxaemia of pregnancy.
                    if line.strip() == '':
                        continue
                    # if len(line.strip().split('\t')) < 4:
                    #     print('!!!')
                    # _, pmid, mesh_terms, _ = line.split('\t') # with title in processed and unique_records
                    _, pmid, mesh_terms = line.split('\t')
                    if pmid not in pmid_meshterms:
                        pmid_meshterms[pmid] = mesh_terms
                    else:
                        repeated_txt = '{} repeated in {}-{}\n'.format(pmid, y, m)
                        repeated_output += repeated_txt
                        print(repeated_txt.strip())

    with open(output_path, 'wb') as pmid_date_file:
        pickle.dump(pmid_meshterms, pmid_date_file, protocol=pickle.HIGHEST_PROTOCOL)
    print("{} end".format(datetime.now().strftime("%H:%M:%S")))
    if repeated_output.strip() == '':
        print('No repeated PMIDs')
    else:
        with open('{}.txt'.format(repeated_path), 'w') as fh:
            fh.write(repeated_txt)


def test_open():
    with open('/data/user/copara/code/projects/graphs/objects/pmid_meshterms.pickle', "rb") as file:
        pmid_date = pickle.load(file)
    # print(pmid_date)
    print('!!!')


def test_code():
    my_dict = {'D005276': 28.722443426856522, 'D006309': 63.655768654124856, 'D006311': 7.049137931034483, 'D006320': 62.36130874007563,
               'D006801': 22197.494054746978, 'D012951': -0.07586206896551723, 'D002277': 273.5851195406858, 'D007822': 29.48137703089749,
               'D007830': 85.47896895338882, 'D001826': 249.83642172630445}
    my_dict = {key: my_dict[key] / 10.0 for key in my_dict}
    print('!!')


def difference_nodes_2014_12(file1_path='/home/cineca/PycharmProjects/graphs/output/di_2014_12_pmids_bad.csv',
                             file2_path='/home/cineca/Documents/dataset/PubMed/2022/metrics/degree/2014/degree_2014_12.csv',
                             output_path='/home/cineca/PycharmProjects/graphs/output/diff_2014_12_nodes.csv', delim='\t'):
    pmid_degreeinfo = {}
    disruption_pmids = set()
    with open(file1_path) as disruption_pmids_file:
        for line in disruption_pmids_file:
            disruption_pmids.add(line.strip())
    with open(file2_path) as degree_pmids_file:
        for line in degree_pmids_file:
            if line.strip() == '':
                continue
            parts = line.split(delim)
            if parts[0] not in pmid_degreeinfo:
                pmid_degreeinfo[parts[0]] = line
    degree_pmids = set(pmid_degreeinfo.keys())
    diff = degree_pmids.difference(disruption_pmids)
    output = ''
    for node in diff:
        # output += pmid_degreeinfo[node]
        output += node + '\n'
    with open(output_path, 'w') as output_file:
        output_file.write(output)

def compare_2014_bad1_bad2(file1_path='/home/cineca/PycharmProjects/graphs/output/di_mp_dict_2014_12_bad2_sorted1.csv',
                           file2_path='/home/cineca/PycharmProjects/graphs/output/di_mp_dict_2014_12_bad1_sorted1.csv',
                           output1_path='/home/cineca/PycharmProjects/graphs/output/2014_12_bad1_common.csv',
                           output2_path='/home/cineca/PycharmProjects/graphs/output/2014_12_bad2_common.csv'):
    bad1, bad2 = {}, {}
    print('{} Reading bad1 file'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    with open(file1_path) as bad1_file:
        for line in bad1_file:
            if line.strip() == '':
                continue
            parts = line.split(',')
            if parts[0] not in bad1:
                bad1[parts[0]] = line
    print('{} end. Reading bad2 file'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    with open(file2_path) as bad2_file:
        for line in bad2_file:
            if line.strip() == '':
                continue
            parts = line.split(',')
            if parts[0] not in bad2:
                bad2[parts[0]] = line

    bad1_nodes = set(bad1.keys())
    bad2_nodes = set(bad2.keys())
    common_bad = bad1_nodes.intersection(bad2_nodes)
    print('{} Writing bad1 file'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    output = ''
    for node in common_bad:
        output += bad1[node]
    with open(output1_path, 'w') as output_file:
        output_file.write(output)
    print('{} end. Writing bad2 file'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    output = ''
    for node in common_bad:
        output += bad2[node]
    with open(output2_path, 'w') as output_file:
        output_file.write(output)
    print('{} end'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


def load_object_pickle(main_path='/home/cineca/Documents/dataset/PubMed/2022/disruption_obj/2014/'):
    in_count_path = '{}{}'.format(main_path, '2014_12_in_count.pickle')
    out_count_path = '{}{}'.format(main_path, '2014_12_out_count.pickle')
    node_incoming_outgoing_path = '{}{}'.format(main_path, '2014_12_node_incoming_outgoing.pickle')
    with open(in_count_path, "rb") as file:
        in_count = pickle.load(file)
    print('in_count: {}'.format(len(in_count)))

    with open(out_count_path, "rb") as file:
        out_count = pickle.load(file)
    print('out_count: {}'.format(len(out_count)))

    with open(node_incoming_outgoing_path, "rb") as file:
        node_incoming_outgoing = pickle.load(file)
    print('node_incoming_outgoing: {}'.format(len(node_incoming_outgoing)))
    print('!!!')

# after generated the file of this function, I added manually the main treenodes, as they are not in the file
def get_mh_descriptor_of_tn(input_file_path, output_file_path):
    with open(input_file_path) as descriptor_file:
        mesh_content = descriptor_file.read().split('*NEWRECORD')
    # print(mesh_content)
    mesh_heading = {}
    tree_numbers = {}
    for line in mesh_content:
        line = line.strip()
        if line == '':
            continue
        record = line.split('\n')
        tree_number, mesh_head_str, unq_id = '', '', ''
        multiple_tn = []
        for r in record:
            r = r.split('=')
            if r[0].strip() == 'MH':
                mesh_head_str = r[1].strip()
            elif r[0].strip() == 'UI':
                unq_id = r[1]
            elif r[0].strip() == 'MN':
                tree_number = r[1]
            if tree_number != '':
                multiple_tn.append(tree_number)
                tree_number = ''
        for _tn in multiple_tn:

            if _tn != '' and _tn not in tree_numbers:
                tree_numbers[_tn] = []

            # if tree_number != '' and mesh_head_str != '' and unq_id != '':
            tree_numbers[_tn].append((mesh_head_str, unq_id))
        # if unq_id not in mesh_heading:
        #     mesh_heading[unq_id] = mesh_head_str
        # print(line)
        # print(']]]')

    with open(output_file_path, 'wt') as tn_mesh_headings_file:
        for tn, mh_uid in tree_numbers.items():
            tn_mesh_headings_file.write('{}\t{}\t{}\n'.format(tn.strip(), mh_uid[0][0].strip(), mh_uid[0][1].strip()))

def get_md_mh_ms_an(input_file_path, output_file_path):
    # https://www.nlm.nih.gov/mesh/dtype.html ASCII MeSH Descriptor Data Elements
    with open(input_file_path) as descriptor_file:
        mesh_content = descriptor_file.read().split('*NEWRECORD')
    # print(mesh_content)
    mesh_descriptor = {}
    # tree_numbers = {}
    for line in mesh_content:
        line = line.strip()
        if line == '':
            continue
        record = line.split('\n')
        mesh_head_str, mesh_annotation, mesh_scope_note, mesh_ui = '', '', '', ''
        # multiple_tn = []
        for r in record:
            r = r.split('=')
            if r[0].strip() == 'MH':
                mesh_head_str = r[1].strip()
            elif r[0].strip() == 'AN':
                mesh_annotation = r[1]
            elif r[0].strip() == 'MS':
                mesh_scope_note = r[1]
            elif r[0].strip() == 'UI':
                mesh_ui = r[1]
        if mesh_ui not in mesh_descriptor:
            mesh_descriptor[mesh_ui] = []
        mesh_descriptor[mesh_ui].append(mesh_head_str)
        mesh_descriptor[mesh_ui].append(mesh_scope_note)
        mesh_descriptor[mesh_ui].append(mesh_annotation)

    # md_mh_ms_an_txt = ''
    my_list = []
    for mh_uid, mh_ms_an in mesh_descriptor.items():
        # md_mh_ms_an_txt += '{}\t{}\t{}\t{}\n'.format(mh_uid.strip(), mh_ms_an[0], mh_ms_an[1], mh_ms_an[2])
        my_list.append({'descriptor': mh_uid.strip(), 'heading': mh_ms_an[0], 'scope_note': mh_ms_an[1], 'annotation': mh_ms_an[2]})
    my_df = pd.DataFrame(my_list)
    # df = pd.read_csv(my_df, delimiter='\t')
    my_df.to_csv(output_file_path, index=False)

    # with open(output_file_path, 'wt') as md_mh_ms_an_file:
    #     for mh_uid, mh_ms_an in mesh_descriptor.items():
    #         md_mh_ms_an_file.write('{}\t{}\t{}\t{}\n'.format(mh_uid.strip(), mh_ms_an[0], mh_ms_an[1], mh_ms_an[2]))
    print('----------------')



def test_pickle_objects(pmid_meshterms_file):
    start_time = datetime.now()
    print('{} Loading pmid_meshterms pickle'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    with open(pmid_meshterms_file, "rb") as file:
        pmid_meshterms = pickle.load(file)
    end_time = datetime.now()
    print('{} end, lasted {}'.format(end_time.strftime("%H:%M:%S"), end_time - start_time))
    c_papers_mt = 0
    for pmid, mts in pmid_meshterms.items():
        if mts.strip() != '':
            c_papers_mt += 1
    print(len(pmid_meshterms), c_papers_mt)
    print('----------------')

def main_mh_text():
    years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    for yy in years:
        print(yy)
        get_md_mh_ms_an('/home/cineca/Documents/dataset/MeSH_descriptor_data/d{}.bin'.format(yy),
                        '/home/cineca/Documents/dataset/MeSH_descriptor_data/md_mh_ms_an_{}.csv'.format(yy))

def main():
    # get_tree_numbers_mesh('/home/cineca/Documents/dataset/MeSH_descriptor_data/d2022.bin',
    #                       '/home/cineca/Documents/dataset/MeSH_descriptor_data/md_treenumbers_2022.txt')
    # for yy in [2014, 2015, 2016, 2017, 2018, 2019, 2020]:
    #     get_tree_numbers_mesh('/home/cineca/Documents/dataset/MeSH_descriptor_data/d{}.bin'.format(yy),
    #                           '/home/cineca/Documents/dataset/MeSH_descriptor_data/md_treenumbers_{}.txt'.format(yy))

    # get_mesh_heading_bin_mesh('/home/cineca/Documents/dataset/MeSH_descriptor_data/d2022.bin',
    #                           '/home/cineca/Documents/dataset/MeSH_descriptor_data/md_meshheadings_2022.txt')
    # get_mesh_heading_bin_mesh('/home/cineca/Documents/dataset/MeSH_descriptor_data/d2014.bin',
    #                           '/home/cineca/Documents/dataset/MeSH_descriptor_data/md_meshheadings_2014.txt')

    # get_mh_descriptor_of_tn('/home/cineca/Documents/dataset/MeSH_descriptor_data/d2022.bin',
    #                         '/home/cineca/Documents/dataset/MeSH_descriptor_data/md_tn_mh_uid_2022.txt')
    # create_pmid_meshterms(main_path='/data/user/copara/dataset/PubMed/2022/unique_records/',
    #                       output_path='/data/user/copara/code/projects/graphs/objects/pmid_meshterms.pickle',
    #                       repeated_path='/data/user/copara/code/projects/graphs/objects/pmid_meshterms_repeated.txt',
    #                       year_s=1781, year_e=2021, month_s=1, month_e=12)

    # ### create pmid_meshterms object per PubMed year
    # years = [2018, 2020]  #  2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
    # for yy in years:
    #     unique_records_path = '/home/cineca/Documents/dataset/PubMed/{}/unique_records/'.format(yy)
    #     years_dirs = os.listdir(unique_records_path)
    #     years_dirs = list(map(int, years_dirs))
    #     yy_min = min(years_dirs)
    #     yy_max = max(years_dirs)
    #     objects_path = '/home/cineca/Documents/dataset/PubMed/{}/objects/'.format(yy)
    #     if not os.path.exists(objects_path):
    #         os.mkdir(objects_path)
    #     create_pmid_meshterms(main_path=unique_records_path,
    #                           output_path='{}/pmid_meshterms.pickle'.format(objects_path),
    #                           repeated_path='{}/pmid_meshterms_repeated.txt'.format(objects_path),
    #                           year_s=yy_min, year_e=yy_max, month_s=1, month_e=12)
    # ### END pmid_meshterms object per PubMed year
    years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    for yy in years:
        print(yy)
        test_pickle_objects('/home/cineca/Documents/dataset/PubMed/{}/objects/pmid_meshterms.pickle'.format(yy))

    # test_open()
    # test_code()
    # difference_nodes_2014_12()
    # load_object_pickle(main_path='/data/user/copara/dataset/PubMed/2022/disruption_obj/2014/')
    # load_object_pickle()

    # load_object_pickle(main_path='/home/copara/dataset/PubMed/2022/disruption_obj/2014/') ####

    # difference_nodes_2014_12(file1_path = '/home/cineca/PycharmProjects/graphs/output/diff_2014_12_nodesonly.csv',
    #                          file2_path = '/home/cineca/PycharmProjects/graphs/output/di_mp_dict_2014_12_debug.csv',
    #                          output_path = '/home/cineca/PycharmProjects/graphs/output/diff_core0_diffnodes.csv',delim=',')
    # compare_2014_bad1_bad2()

# from disruption.di_mcn_dict_mp import create_monthly_objects
import disruption.disruption_utils as disruption_utils
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


def main_disruption():
    # # running in Goldorak: tm_ref2
    # year_s, year_e, month_s, month_e = 2014, 2014, 3, 12
    # # year, month = 1961, 1
    # for year in range(year_s, year_e + 1):
    #     for month in range(month_s, month_e + 1):
    #         main_folder_path = '/data/user/copara/dataset/PubMed/2022'
    #         # main_folder_path = '/home/cineca/Documents/dataset/PubMed/2022'#'/home/cineca/PycharmProjects/graphs/output'
    #         main_process(year=year, month=month, main_folder_path=main_folder_path)


    # Goldorak
    # main_folder_path = '/data/user/copara/dataset/PubMed/2022'
    # destination_folder = '/data/user/copara/dataset/PubMed/2022/disruption_obj'

    # Snowden
    # main_folder_path = '/home/copara/dataset/PubMed/2022'
    # destination_folder = '/home/copara/dataset/PubMed/2022/disruption_obj'

    # Baobab
    main_folder_path = '/home/users/c/coparaz9/dataset/PubMed/2022'
    destination_folder = '{}/disruption_obj'.format(main_folder_path)

    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)

    year_s, year_e, month_s, month_e = 2010, 2010, 1, 1
    # destination_folder = '/home/copara/dataset/PubMed/2022/disruption_obj'
    print('Creating monthly objects for disruption computation')
    for year in range(year_s, year_e + 1):
        destination_folder = '{}/{}'.format(destination_folder, year)
        for month in range(month_s, month_e + 1):
            month = '0' + str(month) if len(str(month)) == 1 else month
            path_nx_graph = '{}/cit_net/{}/cit_net_{}_{}.pickle'.format(main_folder_path, year, year, month)
            if not os.path.exists(path_nx_graph):
                print('{} does not exist in disk'.format(path_nx_graph))
                return
            create_monthly_objects(graph_path=path_nx_graph, output_path=destination_folder, year=year, month=month)

if __name__=='__main__':
    # main()
    # main_disruption()
    main_mh_text()