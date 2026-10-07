from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch_geometric.datasets import Flickr, Planetoid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.models import NodeClassifier
from common.runtime import configure_cpu, elapsed, make_neighbor_loader, save_result


def load_dataset(name: str, root: str):
    name = name.lower()
    if name in {"cora", "citeseer"}:
        dataset_name = "CiteSeer" if name == "citeseer" else "Cora"
        dataset = Planetoid(root=root, name=dataset_name)
        data = dataset[0]
    elif name == "flickr":
        dataset = Flickr(root=root)
        data = dataset[0]
        if not all(hasattr(data, key) for key in ("train_mask", "val_mask", "test_mask")):
            permutation = torch.randperm(data.num_nodes)
            train_end = int(0.6 * data.num_nodes)
            val_end = int(0.8 * data.num_nodes)
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[permutation[:train_end]] = True
            data.val_mask[permutation[train_end:val_end]] = True
            data.test_mask[permutation[val_end:]] = True
    else:
        raise ValueError(f"unsupported dataset: {name}")
    for key in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, key)
        if mask.ndim > 1:
            setattr(data, key, mask[:, 0])
    return data, dataset.num_classes


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    prediction = model(data.x, data.edge_index).argmax(dim=-1)
    return (prediction[mask] == data.y[mask]).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description="CPU node classification on Cora/CiteSeer/Flickr")
    parser.add_argument("--dataset", choices=["Cora", "CiteSeer", "Flickr"], default="Cora")
    parser.add_argument("--model", choices=["gcn", "gat", "sage", "gin"], default="gcn")
    parser.add_argument("--mode", choices=["full", "sampled"], default="full")
    parser.add_argument("--root", default=str(ROOT / "task1_node_classification" / "data"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fanout", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output")
    args = parser.parse_args()
    device = configure_cpu(args.seed, args.threads)
    data, classes = load_dataset(args.dataset, args.root)
    data = data.to(device)
    model = NodeClassifier(
        args.model, data.num_features, args.hidden, classes, args.layers
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sampler_config = None
    if args.mode == "sampled":
        sampler_config = {
            "input_nodes": data.train_mask,
            "num_neighbors": [args.fanout] * args.layers,
            "batch_size": args.batch_size,
            "shuffle": True,
        }

    train_seconds = 0.0
    started = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        epoch_started = time.perf_counter()
        if args.mode == "full":
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
        else:
            loader = make_neighbor_loader(data, **sampler_config)
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch.x, batch.edge_index)[: batch.batch_size]
                loss = F.cross_entropy(logits, batch.y[: batch.batch_size])
                loss.backward()
                optimizer.step()
        train_seconds += time.perf_counter() - epoch_started
    test_accuracy = evaluate(model, data, data.test_mask)
    validation_accuracy = evaluate(model, data, data.val_mask)
    result = {
        "task": "node_classification",
        "dataset": args.dataset,
        "model": args.model,
        "mode": args.mode,
        "layers": args.layers,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "test_accuracy": round(test_accuracy, 6),
        "validation_accuracy": round(validation_accuracy, 6),
        "train_seconds": round(train_seconds, 3),
        "total_seconds": elapsed(started),
        "device": "cpu",
        "sampler": sampler_config,
    }
    save_result(result, args.output)


if __name__ == "__main__":
    main()
