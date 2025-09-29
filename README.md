README

File 'requirements.txt' contains the needed libraries in a Python environment.

Folder data and output are needed. You can find them in this link (uncompress folders and place them in the root folder):
https://drive.google.com/drive/folders/1S9osbc3CHWEwUDTZzIHZpX2KjHWtL8BO?usp=sharing

To compute the metrics (category utility, centrality, disruption, entropy), you need to run:
python sample_main.py

To compute the propagation of the graph-based metrics (centrality and disruption):
Centrality: 
cd centralities
python centralities_concepts.py

Disruption:
cd disruption
python disruption_concepts.py


Folder 'data' contains:
- Folder 'cit_net_sample': citation networks of January and February of 2014.
- Folder 'disruption_obj_sample': needed files to compute disruption for these months.
- Folder 'objects': A pickle file of a dictionary containing PMIDs and MeSH terms (annotations in PubMed articles) of PubMed 2014. The 'md_treenumbers_2014.txt' file contains MeSH descriptors and tree numbers (from the MeSH hierarchy) of MeSH 2014.

Folder 'output' contains:
- Console output of the computation of metrics, propagation of disruption, and centrality.
- Folder 'metrics_sample': 
	- CU/2014/hierarchy_cu_sample_2014_01.csv: contains the category utility value of tree nodes of this month.
	- disruption/2014/di_mp_dict_2014_01.csv: contains the disruption values per article of this month.
	- disruption/2014/treenode_disruption_2014_01.csv: contains the disruption value of tree nodes of this month.
	- entropy/2014/hierarchy_entropy_2014_01.csv: contains the entropy value of tree nodes of this month.
	- pagerank/2014/pagerank_2014_01.csv: contains the centrality (pagerank) values per article of this month.
	- pagerank/2014/treenode_pagerank_2014_01.csv: contains the centrality (pagerank) value of tree nodes of this month.
