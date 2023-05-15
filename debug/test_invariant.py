from utils.common_imports import *
from utils.common_imports_experiment import *
from plane.models import *
from utils.processing.data_process import *
import itertools
import random

model = ModelGraph(
    ModelConfig(
            dim = 32,
            dim_plane_pe = 4,
            num_layers = 3,
            dim_node_feature = [32],
            dim_edge_feature = [4],
            dim_output = 1,
            flags_layer="plane",
            flags_plane_agg="n_t_b_gr_cr",
            flags_mlp_factor=2,
            drop_agg=0,
            drop_com=0,
            drop_enc=0,
            drop_out=0,
            drop_rec=0
    )
)
model.eval()

def test(edge_index, permute_maps):
    num_node = max(max(x) for x in edge_index) + 1
    num_edge = len(edge_index)

    n_permute = len(permute_maps)
    processed_datas = []
    for m in permute_maps:
        g_torch = tgdata.Data(
            x=torch.zeros(num_node, 1, dtype=torch.long),
            edge_index=torch.tensor([
                [m[u],m[v]]
                for uu,vv in edge_index
                for [u,v] in [[uu,vv],[vv,uu]]
            ], dtype=torch.long).mT,
            edge_attr=torch.zeros(num_edge*2, 1, dtype=torch.long),
        )
        processed_datas.append(process(g_torch))
    train_loader = tgloader.DataLoader(processed_datas, batch_size=n_permute, shuffle=False)
    batch = next(iter(train_loader))
    out = model(batch)
    for i in range(n_permute):
        for j in range(n_permute):
            if not torch.allclose(out[i], out[j], atol=1e-5):
                print(out[i])
                print(out[j])
                print("edge_index =",edge_index)
                print("permute_map1 =",permute_maps[i])
                print("permute_map2 =",permute_maps[j])
                raise Exception("Failed")

num_node = 5
num_edge = 7
n_permute = 100

### Generate Graph
planar_iter = sageall.graphs.planar_graphs(
    num_node,
    minimum_edges=num_edge, maximum_edges=num_edge,
    minimum_connectivity=1
)

iter_id = 0
for i,g_sage in enumerate(planar_iter):
    print(i)
    if i>10000: break

    g_sage.relabel({num_node: 0})
    edge_index = list(g_sage.edges(labels=None))

    permute_maps = []
    for _ in range(n_permute):
        m = list(range(num_node))
        random.shuffle(m)
        permute_maps.append(m)

    test(edge_index, permute_maps)