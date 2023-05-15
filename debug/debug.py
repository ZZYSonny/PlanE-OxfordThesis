import time
from utils.common_imports_experiment import *
data = tg.data.Data(
    x = torch.tensor([0]*7),
    y = torch.tensor(0),
    edge_index = torch.tensor([
        [6,5],[5,6],
        [1,5],[5,1],
        [0,1],[1,0],
        [0,3],[3,0],
        [3,2],[2,3],
        [2,1],[1,2],
        [1,3],[3,1],
        [0,2],[2,0],
        [2,4],[4,2],
        [4,3],[3,4],
    ]).mT,
    edge_attr = torch.tensor([0]*20),
)
model_config = models.ModelConfig(
    dim = 32,
    dim_plane_pe = 16,
    num_layers = 3,
    dim_output = 1,
    dim_node_feature=[32],
    dim_edge_feature=4,
    flags_layer="plane",
    flags_plane_agg="n_t_b_gr_cr",
    flags_plane_gine_type="incomplete",
    flags_mlp_factor=-1,
    drop_enc=0,
    drop_rec=0,
    drop_agg=0,
    drop_com=0,
    drop_out=0,
    drop_edg=0
)
device = torch.device("cpu")

model = models.ModelGraph(model_config)
model.to(device)

process = tgtrans.Compose([
    data_process.node_cluster_coefficient_graph,
    data_process.process,
])

trainloader = tgloader.DataLoader([process(data)]*(256*128), batch_size=256, shuffle=True, num_workers=8)



with torch.profiler.profile() as p:
    for batch in tqdm(trainloader):
        batch.to(device)
        out = model(batch)

print(p.key_averages().table(
    sort_by="self_cpu_time_total", row_limit=-1))