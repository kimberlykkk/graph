# 图神经网络入门实验

本项目使用 **PyTorch Geometric (PyG)** 实现节点分类、链路预测、图分类和知识图谱补全，统一支持 CPU。图任务提供全图训练与邻居采样训练两种模式；运行时结果以 JSON 输出，便于比较指标和耗时。

## 环境

建议使用 Python 3.10–3.12。先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)安装与本机匹配的 **CPU 版 PyTorch**，再安装项目依赖：

```powershell
python -m pip install -r requirements.txt
```

NeighborLoader 在不同平台需匹配 PyTorch 版本安装 `pyg-lib`（优先）或 `torch-sparse`。若采样时报扩展缺失，按 [PyG 安装说明](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)选择 CPU wheel。其余全图实验不依赖采样扩展。

默认数据根目录为各任务下的 `data/`，首次运行时 PyG 自动下载数据；不需要预先准备或提交数据集。

## 目录

- `task1_node_classification/`：Cora、CiteSeer、Flickr 节点分类
- `task2_link_prediction/`：Cora、CiteSeer、Flickr 链路预测
- `task3_graph_classification/`：TUDataset 图分类与 ZINC 图回归
- `task4_knowledge_graph/`：TransE、RotatE、ConvE 知识图谱补全
- `common/`：共享模型、CPU 配置和采样工具

各任务目录中的 `README.md` 给出了完整训练命令。所有命令应从仓库根目录运行。

## 模型和实验说明

- **GCN**：按度归一化邻居特征后聚合，再做线性变换。
- **GAT**：学习邻居注意力权重，按权重聚合邻居表示。
- **GraphSAGE**：采样并聚合邻居表示，再与节点表示组合；这里采用 mean 聚合。
- **GIN**：聚合邻居与自身表示，并通过 MLP 更新，具备较强的结构区分能力。
- **TransE**：令头实体向量加关系向量接近尾实体向量。
- **RotatE**：在复数空间中以关系相位旋转头实体向量。
- **ConvE**：卷积头实体和关系表示的组合，再与候选尾实体打分。

节点分类在 Cora/CiteSeer 使用 Planetoid 官方划分，Flickr 使用数据集自带 mask；如缺少 mask，则固定随机拆分训练/验证/测试为 60%/20%/20%。链路预测使用无向 `RandomLinkSplit`，训练图不包含验证或测试正边；负边从原始边集之外采样。图分类对 TU 数据集固定随机拆分 80%/10%/10%；ZINC 使用官方 train/val/test 划分。知识图谱支持分开的 `train/valid/test` 三个文件；传入单文件时以固定随机种子随机拆成 80%/10%/10%。

比较实验时固定随机种子、线程数、数据划分和训练轮数，每次只变更一个因素（模型、学习率、层数、采样模式或池化方法）。输出包含验证/测试指标和训练/总耗时；采样和全图的耗时应在同一台 CPU 上测量。采样实验需要同时注明 fanout、batch size。结果会因 CPU、软件版本和训练轮数而变化，因此本仓库不预填不可复现的性能结论。

分析时可将以下机制作为待数据验证的假设，而不是预设实验结论：GCN 对邻居特征做平滑聚合，层数过多可能导致过平滑；GAT 的注意力能区分邻居贡献，但额外计算注意力会增加耗时；GraphSAGE 的邻居采样适合扩展到大图，fanout 增大通常会提高邻域覆盖和计算成本；GIN 的 MLP 聚合表达力较强，也可能需要更多计算。学习率过高可能使训练不稳定，过低则收敛较慢；增加层数会扩大感受野，同时增加运算量。平均池化关注整体特征，max 池化保留最强响应，min 池化保留最小响应；应在相同数据拆分和训练预算下比较验证集指标与耗时。可以先用学习率 `{0.001, 0.01}`、层数 `{2, 3}` 和全部模型/池化方法做小规模网格，再对较优配置延长训练。

## 结果记录

传入 `--output results/run.json` 可保存完整运行配置与结果。模型、学习率和层数等参数均可从训练脚本的 `--help` 查看。