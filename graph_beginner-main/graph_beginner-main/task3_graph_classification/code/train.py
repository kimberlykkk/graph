from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch_geometric.datasets import TUDataset, ZINC
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.models import GraphClassifier
from common.runtime import configure_cpu, elapsed, save_result


def ensure_node_features(data):
    """Give featureless TU graphs a constant feature and normalize feature dtype."""
    if data.x is None or data.x.numel() == 0:
        data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)
    else:
        data.x = data.x.float()
    return data


def load_datasets(name: str, root: str, seed: int):
    if name.lower() == "zinc":
        datasets = [
            ZINC(root=root, subset=True, split=split, pre_transform=ensure_node_features)
            for split in ("train", "val", "test")
        ]
        return (*datasets, True)
    dataset = TUDataset(root=root, name=name)
    dataset.transform = ensure_node_features
    generator = torch.Generator().manual_seed(seed)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return (*torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=generator), False)


@torch.no_grad()
def evaluate(model, loader, device, regression):
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch.x, batch.edge_index, batch.batch)
        predictions.append(output.view(-1) if regression else output)
        targets.append(batch.y.to(torch.float).view(-1) if regression else batch.y.view(-1))
    output = torch.cat(predictions)
    target = torch.cat(targets)
    if regression:
        return F.mse_loss(output, target).sqrt().item()
    return (output.argmax(dim=-1) == target).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description="CPU graph classification on TU or ZINC")
    parser.add_argument("--dataset", default="MUTAG", help="TU dataset name, or ZINC")
    parser.add_argument("--model", choices=["gcn", "gat", "sage", "gin"], default="gcn")
    parser.add_argument("--pooling", choices=["avg", "max", "min"], default="avg")
    parser.add_argument("--root", default=str(ROOT / "task3_graph_classification" / "data"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output")
    args = parser.parse_args()
    device = configure_cpu(args.seed, args.threads)
    train_set, val_set, test_set, regression = load_datasets(args.dataset, args.root, args.seed)
    sample = train_set[0]
    output_channels = 1 if regression else int(train_set.dataset.num_classes if hasattr(train_set, "dataset") else train_set.num_classes)
    model = GraphClassifier(
        args.model,
        sample.num_features,
        args.hidden,
        output_channels,
        args.layers,
        args.pooling,
    ).to(device)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    started = time.perf_counter()
    train_seconds = 0.0
    for _ in range(args.epochs):
        model.train()
        epoch_started = time.perf_counter()
        for batch in train_loader:
            batch = batch.to(device)
            prediction = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y.to(torch.float).view(-1) if regression else batch.y.view(-1)
            loss = F.mse_loss(prediction.view(-1), target) if regression else F.cross_entropy(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_seconds += time.perf_counter() - epoch_started
    result = {
        "task": "graph_regression" if regression else "graph_classification",
        "dataset": args.dataset,
        "model": args.model,
        "pooling": args.pooling,
        "layers": args.layers,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "validation_rmse" if regression else "validation_accuracy": round(
            evaluate(model, val_loader, device, regression), 6
        ),
        "test_rmse" if regression else "test_accuracy": round(
            evaluate(model, test_loader, device, regression), 6
        ),
        "train_seconds": round(train_seconds, 3),
        "total_seconds": elapsed(started),
        "device": "cpu",
    }
    save_result(result, args.output)


if __name__ == "__main__":
    main()
