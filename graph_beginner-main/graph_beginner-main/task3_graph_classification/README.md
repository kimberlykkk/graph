# 任务三：图分类

支持 TU 数据集和 ZINC。TU 数据集默认按 80%/10%/10% 划分训练、验证和测试集；ZINC 使用数据集自带划分。支持 GCN、GAT、GraphSAGE、GIN，以及平均池化、最大池化和最小池化。

## TU 图分类

```bash
python code/train.py --dataset MUTAG --model gcn --pooling avg --epochs 20 --threads 1 --output results/mutag_gcn_avg.json
```

## ZINC 图回归

```bash
python code/train.py --dataset ZINC --model gcn --pooling avg --epochs 20 --threads 1 --output results/zinc_gcn_avg.json
```

池化参数可选 `avg`、`max`、`min`；TU 数据集输出准确率，ZINC 输出 RMSE。
