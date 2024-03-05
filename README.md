## Taxi Traffic Benchmark

### Pipeline

<img decoding="async" src="./imgs/ttb_pipeline.png" width="" height="">

### Dataset

- [Shanghai Taxi](https://cse.hkust.edu.hk/scrg/)

### Baseline

- ARIMA
- SVR
- LSTM
- STGCN[[Paper]](https://arxiv.org/abs/1709.04875) [[Code]](./models/stgcn) [[Official Code]](https://github.com/hazdzz/STGCN)
- ASTGCN[[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/3881) [[Code]](./models/ASTGCN_r.py) [[Official Code]](https://github.com/guoshnBJTU/ASTGCN-r-pytorch)
- Graph WaveNet[[Paper]](https://arxiv.org/pdf/1906.00121) [[Code]](./models/graph_wavenet.py) [[Official Code]](https://github.com/nnzhan/Graph-WaveNet)

### Proposed

[[Paper]](https://arxiv.org/abs/2401.08727)**MA2GCN: Multi Adjacency relationship Attention Graph Convolutional
Networks for Traffic Prediction using Trajectory data**

| <img decoding="async" src="./imgs/arch.png" width="400" height=""> | <img decoding="async" src="./imgs/adj_attention.png" width="400" height=""> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                         Architecture                         |       Multi Adjacency relationship Attention mechanism       |

### Related Links

- GPS2Graph [Link](https://github.com/zachysun/Gps2graph)

### Important Notes

- The Shanghai Taxi Dataset rarely appears in traffic prediction tasks, so the six baselines above can be used as a reference.
- The io matrix and the vehicle entry and exit matrix in MA2GCN paper have the same meaning. It is transformed into the mobility matrix and used as an input.
- The six baselines can't use the mobility matrix as it's specificly used for proposed model(MA2GCN).
- Due to the mobility matrix is time sensitive, the dataset needs to be divided in chronological order and cannot be shuffled. For the six baselines, there is no such limit. 
- The proposed model is on arxiv.org. Now it can be seen as a simple technical report exploring the use of taxi trajectories for traffic prediction tasks. There's still significant potential for enhancement. 
