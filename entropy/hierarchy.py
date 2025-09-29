

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

def aggregate_bottom_top(tree_leaf_referred: dict, root: str):
    new_tree_leaf_referred = tree_leaf_referred.copy()
    for node, freq in tree_leaf_referred.items():
        num_levels = node.count('.')
        parent_node = node
        for i in range(0, num_levels):
            index_point = parent_node.rfind('.')
            parent_node = parent_node[0:index_point]
            if parent_node in new_tree_leaf_referred:
                new_tree_leaf_referred[parent_node] += freq
            else:
                new_tree_leaf_referred[parent_node] = freq
        # tmp_increase = new_tree_leaf_referred[parent_node[0]] + freq
        # new_tree_leaf_referred[parent_node[0]] = tmp_increase if parent_node[0] in new_tree_leaf_referred else freq
        # to aggregate level 2 (A01, A10, A11, B01, B02...) to level 1 (A, B)
        if parent_node[0] in new_tree_leaf_referred:
            new_tree_leaf_referred[parent_node[0]] += freq
        else:
            new_tree_leaf_referred[parent_node[0]] = freq
    # to include the root in the aggregation: A, B, C, ... counts will be sum into MeSH node
    new_tree_leaf_referred[root] = 0
    for node, acc in new_tree_leaf_referred.items():
        if len(node) == 1:
            # if root not in new_tree_leaf_referred:
            new_tree_leaf_referred[root] += new_tree_leaf_referred[node]

    return new_tree_leaf_referred
