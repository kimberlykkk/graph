# 任务一：节点分类

使用 Cora、CiteSeer 或 Flickr 进行节点分类，支持 GCN、GAT、GraphSAGE 和 GIN。脚本会报告验证集、测试集准确率与训练耗时。

## 全图训练

```bash
python code/train.py --dataset Cora --model gcn --mode full --epochs 20 --threads 1 --output results/cora_gcn_full.json
```

## NeighborLoader 采样训练

```bash
python code/train.py --dataset Cora --model sage --mode sampled --epochs 20 --batch-size 256 --fanout 10 --threads 1 --output results/cora_sage_sampled.json
```

采样模式需要安装与当前 Python、PyTorch 和 CPU 版本匹配的 `pyg-lib` 或 `torch-sparse`。资源有限时可使用 `--mode full`、较小的 `--hidden` 和较少的 `--epochs`。
