# 任务四：知识图谱补全

支持 TransE、RotatE 和轻量 ConvE 风格模型，使用负采样训练，并用 filtered MRR 和 Hits@10 评估。

## 数据格式

`--data` 可以指向一个目录，目录中包含：

```text
train.tsv
valid.tsv
test.tsv
```

每行三个字段：`head<TAB>relation<TAB>tail`。也支持空格、逗号分隔。若只提供一个文件，脚本会按 8:1:1 随机划分。

## 训练与测试

```bash
python code/train.py --data data/my_kg --model transe --dim 100 --epochs 20 --threads 1 --output results/transe.json
python code/train.py --data data/my_kg --model rotate --dim 100 --epochs 20 --threads 1
python code/train.py --data data/my_kg --model conve --dim 100 --epochs 20 --threads 1
```

CPU 运行时可降低 `--dim`、`--batch-size` 和 `--epochs`。RotatE 的 `--dim` 必须为偶数。

项目自带的 `data/example/` 可用于快速验证三个模型：

```bash
python code/train.py --data data/example --model transe --dim 8 --epochs 2 --threads 1
```
