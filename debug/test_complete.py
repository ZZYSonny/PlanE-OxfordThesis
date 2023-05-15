from utils.common_imports import *
from utils.common_imports_experiment import *
from plane.models import *
from utils.processing.data_process import *
import itertools
import random

def faketqdm(iterable, *args, **kwargs):
    return iterable

def test(models, graphs):
    model = models[0]
    processed_datas = []
    for g in tqdm(graphs, desc="Process Graph"):
        g_torch = tgdata.Data(
            x=torch.zeros(g.num_verts(), 1, dtype=torch.long),
            edge_index=torch.tensor([
                [u,v]
                for uu,vv in g.edge_iterator(labels=False)
                for [u,v] in [[uu,vv],[vv,uu]]
            ], dtype=torch.long).mT,
            edge_attr=torch.zeros(g.num_edges()*2, 1, dtype=torch.long),
        )
        processed_datas.append(process(g_torch))

    train_loader = tgloader.DataLoader(processed_datas, batch_size=64, shuffle=False)
    out_batch = []
    for batch in tqdm(train_loader, desc="Eval Graph"):
        out_batch.append(model(batch))
    out = torch.cat(out_batch, dim=0)

    for i in tqdm(range(len(graphs)), desc="Compare Graph", leave=False):
        fail_graphs = [graphs[i]]
        for j in range(i+1, len(graphs)):
            if torch.allclose(out[i], out[j], atol=1e-5):
                if not graphs[i].is_isomorphic(graphs[j]):
                    if len(models) > 1:
                        fail_graphs.append(graphs[j])
                    else:
                        print("Find counterexample.")
                        print(out[i])
                        print(out[j])
                        print("edge_index1 =",list(graphs[i].edge_iterator(labels=False)))
                        print("edge_index2 =",list(graphs[j].edge_iterator(labels=False)))
                        raise Exception("Failed")
                
        if len(fail_graphs) >1:
            test(models[1:], fail_graphs)


num_node = 8
num_edge = 12
max_graphs = 10000

flags_add = "n_t_gr"
flags_compute = ""

models = [
    ModelGraph(ModelConfig(
        dim = 32,
        dim_pe = 16,
        num_layers = 3,
        dim_node_feature=[32],
        dim_edge_feature=[4],
        dim_output = 1,
        flags_plane_agg=flags_add,
        flags_compute=flags_compute,
    )),
    ModelGraph(ModelConfig(
        dim = 32,
        dim_pe = 16,
        num_layers = 4,
        dim_node_feature=[32],
        dim_edge_feature=[4],
        dim_output = 1,
        flags_plane_agg=flags_add,
        flags_compute=flags_compute,
    ))
]
for m in models:
    m.eval()

### Generate Graph
planar_graphs = list(itertools.islice(
    sageall.graphs.planar_graphs(
        num_node,
        minimum_edges=num_edge, maximum_edges=num_edge,
        minimum_connectivity=1
    ), max_graphs
))
for g in planar_graphs:
    g.relabel({num_node: 0})
test(models, planar_graphs)
