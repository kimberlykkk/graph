# 任务二：链路预测

使用 Cora、CiteSeer 或 Flickr 预测节点对之间是否存在边。数据由 `RandomLinkSplit` 划分为训练、验证和测试集，脚本报告 AUC、准确率和耗时。

## 全图训练

```bash
python code/train.py --dataset Cora --model gcn --mode full --epochs 20 --threads 1 --output results/cora_gcn_full.json
```

## 采样训练

```bash
python code/train.py --dataset Cora --model sage --mode sampled --epochs 20 --batch-size 512 --fanout 10 --threads 1 --output results/cora_sage_sampled.json
```

采样模式需要匹配当前 Python、PyTorch 和 CPU 版本的 `pyg-lib` 或 `torch-sparse`；CPU 资源有限时建议先运行全图模式。
