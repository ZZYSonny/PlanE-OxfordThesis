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


model1 = ModelGraph(
    ModelConfig(
            dim = 16,
            dim_pe = 8,
            drop_mp= 0,
            drop_read= 0,
            num_layers = 4,
            dim_output=1,
            dim_node_feature=[32],
            dim_edge_feature=[4],
            b_mp_encoder=False
    )
)
model1.eval()

edge_index1 = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (3, 7), (5, 6), (5, 7), (0, 5), (0, 6)]
edge_index2 = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 4), (3, 6), (3, 7), (4, 6), (0, 4), (0, 5)]

### Generate Graph
planar_graphs = [
    sageall.Graph(edge_index1),
    sageall.Graph(edge_index2)
]
test([model1], planar_graphs)
