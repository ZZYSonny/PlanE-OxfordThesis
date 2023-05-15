from utils.common_imports import *
from utils.common_imports_experiment import *
from plane.models import *
from utils.processing.data_process import *

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
    model = ModelGraph(
        ModelConfig(
            dim = 8,
            dim_plane_pe = 4,
            num_layers = 2,
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
    train_loader = tgloader.DataLoader(processed_datas, batch_size=n_permute, shuffle=False)
    batch = next(iter(train_loader))
    out = model(batch)
    for i in range(n_permute):
        for j in range(n_permute):
            if not torch.allclose(out[i], out[j], atol=1e-5):
                print(f"Failed {i} {j}")
                print(out[i].tolist())
                print(out[j].tolist())

edge_index = [(0, 3), (0, 4), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
permute_map1 = [4, 1, 0, 2, 3]
permute_map2 = [1, 0, 3, 2, 4]


permute_maps = [
    permute_map1,
    permute_map2
]

test(edge_index,permute_maps)