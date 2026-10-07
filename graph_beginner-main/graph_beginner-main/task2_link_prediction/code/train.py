from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch_geometric.datasets import Flickr, Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.models import GNNEncoder
from common.runtime import configure_cpu, elapsed, make_neighbor_loader, save_result


def load_dataset(name: str, root: str):
    name = name.lower()
    if name in {"cora", "citeseer"}:
        dataset_name = "CiteSeer" if name == "citeseer" else "Cora"
        original = Planetoid(root=root, name=dataset_name)[0]
    elif name == "flickr":
        original = Flickr(root=root)[0]
    else:
        raise ValueError(f"unsupported dataset: {name}")
    original = original.cpu()
    all_edges = original.edge_index.clone()
    transform = RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        disjoint_train_ratio=0.2,
    )
    train_data, val_data, test_data = transform(original)
    return train_data, val_data, test_data, all_edges


def auc_score(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.cpu().long()
    scores = scores.cpu()
    positives = int(labels.sum())
    negatives = labels.numel() - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    rank_sum = 0.0
    offset = 0
    for count in counts.tolist():
        rank = offset + (count + 1) / 2
        rank_sum += rank * float(sorted_labels[offset : offset + count].sum())
        offset += count
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


@torch.no_grad()
def evaluate(model, data, edges):
    model.eval()
    z = model(data.x, data.edge_index)
    scores = (z[edges.edge_label_index[0]] * z[edges.edge_label_index[1]]).sum(dim=-1)
    labels = edges.edge_label
    prediction = scores >= 0
    accuracy = (prediction == labels.bool()).float().mean().item()
    return auc_score(labels, scores), accuracy


def train_sampled(model, optimizer, data, positive_edges, negative_edges, args):
    combined = torch.cat((positive_edges, negative_edges), dim=1)
    labels = torch.cat(
        (
            torch.ones(positive_edges.size(1)),
            torch.zeros(negative_edges.size(1)),
        )
    )
    permutation = torch.randperm(combined.size(1))
    for positions in permutation.split(args.batch_size):
        edge_batch = combined[:, positions]
        y = labels[positions]
        seed_nodes = torch.unique(edge_batch.reshape(-1))
        loader = make_neighbor_loader(
            data,
            input_nodes=seed_nodes,
            num_neighbors=[args.fanout] * args.layers,
            batch_size=seed_nodes.numel(),
            shuffle=False,
        )
        batch = next(iter(loader)).to(data.x.device)
        local_index = torch.full(
            (data.num_nodes,), -1, dtype=torch.long, device=data.x.device
        )
        local_index[batch.n_id] = torch.arange(batch.n_id.numel(), device=data.x.device)
        local_edges = local_index[edge_batch.to(data.x.device)]
        z = model(batch.x, batch.edge_index)
        logits = (z[local_edges[0]] * z[local_edges[1]]).sum(dim=-1)
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits, y.to(data.x.device))
        loss.backward()
        optimizer.step()


def main():
    parser = argparse.ArgumentParser(description="CPU link prediction on Cora/CiteSeer/Flickr")
    parser.add_argument("--dataset", choices=["Cora", "CiteSeer", "Flickr"], default="Cora")
    parser.add_argument("--model", choices=["gcn", "gat", "sage", "gin"], default="gcn")
    parser.add_argument("--mode", choices=["full", "sampled"], default="full")
    parser.add_argument("--root", default=str(ROOT / "task2_link_prediction" / "data"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512, help="supervised links per batch")
    parser.add_argument("--fanout", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output")
    args = parser.parse_args()
    device = configure_cpu(args.seed, args.threads)
    train_data, val_data, test_data, all_edges = load_dataset(args.dataset, args.root)
    train_data = train_data.to(device)
    val_data, test_data = val_data.to(device), test_data.to(device)
    all_edges = all_edges.to(device)
    model = GNNEncoder(args.model, train_data.num_features, args.hidden, args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_positive = train_data.edge_label_index.to(device)
    train_negative = negative_sampling(
        all_edges, num_nodes=train_data.num_nodes, num_neg_samples=train_positive.size(1)
    ).to(device)

    started = time.perf_counter()
    train_seconds = 0.0
    for _ in range(args.epochs):
        model.train()
        epoch_started = time.perf_counter()
        if args.mode == "full":
            z = model(train_data.x, train_data.edge_index)
            positive_logits = (z[train_positive[0]] * z[train_positive[1]]).sum(dim=-1)
            negative_logits = (z[train_negative[0]] * z[train_negative[1]]).sum(dim=-1)
            logits = torch.cat((positive_logits, negative_logits))
            labels = torch.cat((torch.ones_like(positive_logits), torch.zeros_like(negative_logits)))
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            train_sampled(model, optimizer, train_data, train_positive, train_negative, args)
        train_seconds += time.perf_counter() - epoch_started
    validation_auc, validation_accuracy = evaluate(model, train_data, val_data)
    test_auc, test_accuracy = evaluate(model, train_data, test_data)
    result = {
        "task": "link_prediction",
        "dataset": args.dataset,
        "model": args.model,
        "mode": args.mode,
        "layers": args.layers,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "validation_auc": round(validation_auc, 6),
        "validation_accuracy": round(validation_accuracy, 6),
        "test_auc": round(test_auc, 6),
        "test_accuracy": round(test_accuracy, 6),
        "train_seconds": round(train_seconds, 3),
        "total_seconds": elapsed(started),
        "device": "cpu",
    }
    save_result(result, args.output)


if __name__ == "__main__":
    main()
