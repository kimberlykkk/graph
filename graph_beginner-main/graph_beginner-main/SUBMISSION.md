## Install

```powershell
$python = "C:\Users\there\AppData\Local\Programs\Python\Python312\python.exe"
& $python -m pip install -r requirements.txt
```

## Run each task

Run these commands from the repository root. Increase `--epochs` for an actual training experiment and retain the output JSON with the submitted results.

```powershell
$python = "C:\Users\there\AppData\Local\Programs\Python\Python312\python.exe"
& $python task1_node_classification/code/train.py --dataset Cora --model gcn --mode full --epochs 20 --threads 1 --output results/task1_cora_gcn.json
& $python task2_link_prediction/code/train.py --dataset Cora --model gcn --mode full --epochs 20 --threads 1 --output results/task2_cora_gcn.json
& $python task3_graph_classification/code/train.py --dataset MUTAG --model gcn --pooling avg --epochs 20 --threads 1 --output results/task3_mutag_gcn.json
& $python task4_knowledge_graph/code/train.py --data task4_knowledge_graph/data/example --model transe --dim 8 --epochs 20 --threads 1 --output results/task4_example_transe.json
```

## Existing verification records

The JSON files in `results/` record short CPU execution checks, not converged or comparable benchmark results. Task 3's recorded check uses generated local graphs rather than MUTAG; the MUTAG download did not complete. Task 4 uses the bundled four-entity example graph, so its ranking metrics are only useful for confirming execution. Do not present these records as final model-quality results.

Commands and captured text output for the existing checks are documented in [results/RUN_OUTPUTS.md](results/RUN_OUTPUTS.md). No authentic desktop-terminal screenshot was captured; the report is a text record, not an image of a terminal window.
