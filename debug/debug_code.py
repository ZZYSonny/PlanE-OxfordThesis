from utils.common_imports import *
from utils.common_imports_experiment import *
from utils.processing.data_process import *
pwl_iter=2


def test(edge_index, permute_map):
    #inv_permute_map = list(range(len(permute_map)))
    inv_permute_map = [permute_map.index(i) for i in range(len(permute_map))]
    num_nodes = max(max(x) for x in edge_index) + 1

    g_sage = sageall.Graph([
        (permute_map[u], permute_map[v])
        for uu,vv in edge_index
        for u,v in [(uu,vv),(vv,uu)]
    ])

    node_map = {i: 0 for i in range(num_nodes)}
    edge_map = {(permute_map[u], permute_map[v], None): 0 for uu,vv in edge_index for u,v in [(uu,vv),(vv,uu)]}

    tree, bcc_tree_and_code = data_process_classical.encode_cc(g_sage, node_map, edge_map, pwl_iter=pwl_iter)

    for bcc_nodes, (bcc_spqr_tree, center, code) in bcc_tree_and_code.items():
        print('B', [inv_permute_map[i] for i in bcc_nodes])
        print(code.code)
        print()
        for (sub_type, sub_graph), (spqr_cycle, spqr_code) in code.get_cycles().items():
            print(sub_type, [inv_permute_map[i] for i in list(sub_graph.vertex_iterator())])
            print(spqr_code)
            for (u,v,label) in spqr_cycle:
                print(inv_permute_map[u], inv_permute_map[v])
            print()
        print()

edge_index = [(0, 3), (0, 4), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
permute_map1 = [4, 1, 0, 2, 3]
permute_map2 = [1, 0, 3, 2, 4]

test(edge_index,permute_map1)
print("-------------------")
test(edge_index,permute_map2)

#edge_index1 = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7), (3, 4), (3, 7), (0, 4), (5, 6), (0, 5), (6, 7), (0, 6), (0, 7)]
#edge_index2 = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 6), (3, 4), (3, 6), (4, 7), (5, 7), (0, 5), (6, 7), (0, 6), (0, 7)]
#num_nodes = max(max(x) for x in edge_index1) + 1
#permute_map = list(range(num_nodes))
#
#test(edge_index1,permute_map)
#print("-------------------")
#test(edge_index2,permute_map)

#edge_index = [
#            [0,1],[1,0],
#            [1,3],[3,1],
#            [3,2],[2,3],
#            [2,0],[0,2],
#            [0,3],[3,0],
#            [1,2],[2,1],
#            [2,4],[4,2],
#            [3,4],[4,3],
#            [1,5],[5,1],
#            [5,6],[6,5]
#        ]
#num_nodes = max(max(x) for x in edge_index) + 1
#permute_map = list(range(num_nodes))
#
#test(edge_index,permute_map)