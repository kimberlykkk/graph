from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.models import KnowledgeGraphScorer
from common.runtime import configure_cpu, elapsed, save_result


def read_triples(path: Path) -> list[tuple[str, str, str]]:
    triples = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", "\t").split()
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_number}: expected head, relation, tail")
            if line_number == 1 and parts[0].lower() in {"head", "subject"}:
                continue
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def sample_negative_batch(
    batch: torch.Tensor, entities: int, known_triples: set[tuple[int, int, int]]
) -> torch.Tensor:
    """Corrupt one endpoint while avoiding every known positive triple."""
    negatives = batch.clone()
    for row in range(batch.size(0)):
        head, relation, tail = (int(value) for value in batch[row].tolist())
        for _ in range(32):
            if random.random() < 0.5:
                candidate = (random.randrange(entities), relation, tail)
                if candidate[0] != head and candidate not in known_triples:
                    negatives[row, 0] = candidate[0]
                    break
            else:
                candidate = (head, relation, random.randrange(entities))
                if candidate[2] != tail and candidate not in known_triples:
                    negatives[row, 2] = candidate[2]
                    break
    return negatives


def load_splits(path: str, seed: int):
    source = Path(path)
    if source.is_dir():
        def locate(stem):
            for suffix in (".tsv", ".txt", ".csv"):
                candidate = source / f"{stem}{suffix}"
                if candidate.is_file():
                    return candidate
            raise FileNotFoundError(f"could not find {stem}.tsv/.txt/.csv in {source}")

        train = read_triples(locate("train"))
        valid = read_triples(locate("valid")) if any((source / f"valid{ext}").is_file() for ext in (".tsv", ".txt", ".csv")) else []
        test = read_triples(locate("test")) if any((source / f"test{ext}").is_file() for ext in (".tsv", ".txt", ".csv")) else []
    elif source.is_file():
        triples = read_triples(source)
        random.Random(seed).shuffle(triples)
        train_end = int(0.8 * len(triples))
        valid_end = int(0.9 * len(triples))
        train, valid, test = triples[:train_end], triples[train_end:valid_end], triples[valid_end:]
    else:
        raise FileNotFoundError(f"knowledge graph data not found: {source}")
    if not train or not valid or not test:
        raise ValueError("training, validation, and test splits must all contain triples")
    entities = sorted({node for split in (train, valid, test) for h, _, t in split for node in (h, t)})
    relations = sorted({relation for split in (train, valid, test) for _, relation, _ in split})
    entity_ids = {name: index for index, name in enumerate(entities)}
    relation_ids = {name: index for index, name in enumerate(relations)}

    def encode(split):
        return torch.tensor(
            [(entity_ids[h], relation_ids[r], entity_ids[t]) for h, r, t in split],
            dtype=torch.long,
        )

    return encode(train), encode(valid), encode(test), len(entities), len(relations)


@torch.no_grad()
def evaluate(model, triples, known, entities, device):
    model.eval()
    reciprocal_ranks = []
    hits_at_10 = []
    for head, relation, tail in triples.tolist():
        h = torch.tensor([head], device=device)
        r = torch.tensor([relation], device=device)
        scores = model.score_all_tails(h, r)[0]
        target_score = scores[tail].item()
        for filtered_tail in known.get((head, relation), ()):
            if filtered_tail != tail:
                scores[filtered_tail] = -torch.inf
        rank = 1 + int((scores > target_score).sum())
        reciprocal_ranks.append(1.0 / rank)
        hits_at_10.append(float(rank <= 10))
    return sum(reciprocal_ranks) / len(reciprocal_ranks), sum(hits_at_10) / len(hits_at_10)


def main():
    parser = argparse.ArgumentParser(description="CPU knowledge graph embedding training")
    parser.add_argument("--data", required=True, help="directory with train/valid/test TSV files, or a triples file")
    parser.add_argument("--model", choices=["transe", "rotate", "conve"], default="transe")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output")
    args = parser.parse_args()
    device = configure_cpu(args.seed, args.threads)
    train, valid, test, entities, relations = load_splits(args.data, args.seed)
    train, valid, test = train.to(device), valid.to(device), test.to(device)
    model = KnowledgeGraphScorer(args.model, entities, relations, args.dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    known = {}
    known_triples = {
        (int(head), int(relation), int(tail))
        for head, relation, tail in torch.cat((train, valid, test)).tolist()
    }
    for head, relation, tail in torch.cat((train, valid, test)).tolist():
        known.setdefault((head, relation), set()).add(tail)

    started = time.perf_counter()
    train_seconds = 0.0
    for _ in range(args.epochs):
        model.train()
        epoch_started = time.perf_counter()
        permutation = torch.randperm(train.size(0), device=device)
        for indices in permutation.split(args.batch_size):
            batch = train[indices]
            negatives = sample_negative_batch(batch, entities, known_triples)
            positive_scores = model.score(batch[:, 0], batch[:, 1], batch[:, 2])
            negative_scores = model.score(negatives[:, 0], negatives[:, 1], negatives[:, 2])
            loss = F.softplus(-positive_scores).mean() + F.softplus(negative_scores).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_seconds += time.perf_counter() - epoch_started
    validation_mrr, validation_hits10 = evaluate(model, valid, known, entities, device)
    test_mrr, test_hits10 = evaluate(model, test, known, entities, device)
    result = {
        "task": "knowledge_graph_completion",
        "model": args.model,
        "embedding_dim": args.dim,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "entities": entities,
        "relations": relations,
        "validation_mrr": round(validation_mrr, 6),
        "validation_hits_at_10": round(validation_hits10, 6),
        "test_mrr": round(test_mrr, 6),
        "test_hits_at_10": round(test_hits10, 6),
        "train_seconds": round(train_seconds, 3),
        "total_seconds": elapsed(started),
        "device": "cpu",
    }
    save_result(result, args.output)


if __name__ == "__main__":
    main()
