# CPU Verification Run Record

These are captured outputs from short execution checks on the local CPU environment. They establish that the selected training/evaluation paths ran; they are not final benchmark results. The task 1/2 records use one epoch, task 3 uses synthetic local graphs, and task 4 uses a four-entity example graph.

Environment recorded for these runs: Python 3.12.5, PyTorch 2.14.1+cpu, PyTorch Geometric 2.8.0.post1, CPU, one thread.

## Task 1: Node classification

Command:

```powershell
python task1_node_classification/code/train.py --dataset Cora --model gcn --mode full --epochs 1 --layers 2 --hidden 16 --threads 1
```

Captured result: device `cpu`, validation accuracy `0.100`, test accuracy `0.106`, training time `0.088 s`, total time `0.112 s`.

Machine-readable record: [task1_cora_gcn_cpu_check.json](task1_cora_gcn_cpu_check.json).

## Task 2: Link prediction

Command:

```powershell
python task2_link_prediction/code/train.py --dataset Cora --model gcn --mode full --epochs 1 --layers 2 --hidden 16 --threads 1
```

Captured result: device `cpu`, validation AUC `0.674580`, test AUC `0.674722`, test accuracy `0.500`, training time `0.058 s`, total time `0.114 s`.

Machine-readable record: [task2_cora_gcn_cpu_check.json](task2_cora_gcn_cpu_check.json).

## Task 3: Graph classification pipeline

The local synthetic-graph check completed with loss `0.635485` and accuracy `0.500`. It did not evaluate MUTAG. The official dataset download was interrupted by the remote host, so there is no verified MUTAG result to submit as an experiment.

Machine-readable record: [task3_local_pipeline_check.json](task3_local_pipeline_check.json).

## Task 4: Knowledge graph completion

All three model implementations ran on the bundled example graph (4 entities, 2 relations). Each run is too small to support a model-quality claim; Hits@10 is especially uninformative with only four entities.

| Model | Epochs | Validation MRR | Test MRR | Test Hits@10 |
|---|---:|---:|---:|---:|
| TransE | 2 | 0.500000 | 0.416667 | 1.000000 |
| RotatE | 1 | 0.333333 | 0.416667 | 1.000000 |
| ConvE | 1 | 0.666667 | 0.666667 | 1.000000 |

Machine-readable records: [TransE](task4_transe_example_cpu_check.json), [RotatE](task4_rotate_example_cpu_check.json), [ConvE](task4_conve_example_cpu_check.json).
