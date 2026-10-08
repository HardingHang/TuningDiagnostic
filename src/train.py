"""End-to-end training and minimal functional verification of the learned
tuning diagnostic (Section 4.2 "End-to-End Training", Section 5 metrics).

Usage:  python -m src.train
Checks:
  1. schema graph construction from DDL
  2. query / index-config featurization shapes
  3. end-to-end training reduces loss and beats the all-positive baseline
  4. cross-database: zero-shot on an unseen schema, and fine-tuning a
     pretrained model on a few leaked samples vs training from scratch
"""

import numpy as np
import torch
import torch.nn as nn

from .data_gen import generate_database, generate_dataset, DiagnosticSample
from .featurizer import QUERY_FEATURE_DIM, INDEX_FEATURE_DIM
from .model import TuningDiagnostic


def forward(model: TuningDiagnostic, s: DiagnosticSample) -> torch.Tensor:
    return model(s.queries, s.weights, s.config, s.budget, s.theta)


def train_model(model, samples, epochs, lr=0.01, batch_size=32, seed=0, verbose=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(samples))
        total = 0.0
        for k in range(0, len(order), batch_size):
            batch = order[k:k + batch_size]
            logits = torch.stack([forward(model, samples[i]) for i in batch])
            labels = torch.tensor([float(samples[i].label) for i in batch])
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        if verbose and (epoch + 1) % 2 == 0:
            print(f"    epoch {epoch + 1}: loss={total / len(samples):.4f}", flush=True)
    return total / len(samples)


def evaluate(model, samples, clf_threshold=0.5):
    model.eval()
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for s in samples:
            pred = torch.sigmoid(forward(model, s)).item() >= clf_threshold
            if pred and s.label:
                tp += 1
            elif pred:
                fp += 1
            elif s.label:
                fn += 1
            else:
                tn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1,
            "positive_rate": (tp + fn) / len(samples)}


def main():
    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    print("== 1. Schema graph ==", flush=True)
    db_a = generate_database(rng, prefix="a", n_tables=5)
    g = db_a.graph
    n_tables = sum(g.is_table)
    print(f"  nodes={len(g.nodes)} ({n_tables} tables), edges={len(g.edges)}, "
          f"relations on edges={len({r for _, r, _ in g.edges})}", flush=True)
    assert n_tables == 5 and len(g.edges) > 0

    print("== 2. Dataset (database A) ==", flush=True)
    data_a = generate_dataset(rng, db_a, n_samples=500, thetas=(0.15,))
    pos = sum(s.label for s in data_a)
    print(f"  samples={len(data_a)}, positive rate={pos / len(data_a):.2f}", flush=True)
    s0 = data_a[0]
    assert s0.queries[0].features.shape[1] == QUERY_FEATURE_DIM
    assert s0.config.features.shape[1] == INDEX_FEATURE_DIM
    assert s0.queries[0].features.shape[0] == len(g.nodes)

    perm = np.random.default_rng(0).permutation(len(data_a))
    split = int(0.8 * len(data_a))
    train_a = [data_a[i] for i in perm[:split]]
    test_a = [data_a[i] for i in perm[split:]]

    print("== 3. End-to-end training on A ==", flush=True)
    model = TuningDiagnostic(QUERY_FEATURE_DIM, INDEX_FEATURE_DIM)
    train_model(model, train_a, epochs=200, verbose=True)
    m = evaluate(model, test_a)
    p_base = sum(s.label for s in test_a) / len(test_a)
    f1_base = 2 * p_base / (1 + p_base)
    print(f"  test: precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}",
          flush=True)
    print(f"  all-positive baseline f1={f1_base:.3f}", flush=True)
    assert m["f1"] >= 0.72, f"F1 too low: {m['f1']:.3f}"
    assert m["f1"] > f1_base + 0.05, "not enough margin over the all-positive baseline"

    print("== 4. Cross-database learning (A -> B) ==", flush=True)
    rng_b = np.random.default_rng(7)
    db_b = generate_database(rng_b, prefix="b", n_tables=6)
    assert len(db_b.graph.nodes) != len(g.nodes) or db_b.graph.nodes != g.nodes
    data_b = generate_dataset(rng_b, db_b, n_samples=400, thetas=(0.15,))

    # pretrain on the complete dataset of A (Section 5.3 protocol)
    import copy
    torch.manual_seed(42)
    pre = TuningDiagnostic(QUERY_FEATURE_DIM, INDEX_FEATURE_DIM)
    train_model(pre, data_a, epochs=200, seed=0)

    # 4a. structural transferability: the model trained on A runs zero-shot on
    # B's different schema -- impossible with one-hot representations.
    perm_b = np.random.default_rng(1).permutation(len(data_b))
    held_out_0 = [data_b[i] for i in perm_b[150:]]
    m_zs = evaluate(pre, held_out_0)
    preds = [torch.sigmoid(forward(pre, s)).item() >= 0.5 for s in held_out_0]
    assert 0 < sum(preds) < len(preds), "degenerate zero-shot predictions"
    print(f"  zero-shot on B (different schema): f1={m_zs['f1']:.3f}, "
          f"predictions non-degenerate", flush=True)

    # 4b. small-leak regime (Section 5.3): pretrain on A, fine-tune on a few
    # leaked samples of B, vs training on the leaked data from scratch.
    scores = {"zero-shot": [], "from-scratch": [], "fine-tuned": []}
    for seed in (1, 2, 3):
        perm_b = np.random.default_rng(seed).permutation(len(data_b))
        leaked = [data_b[i] for i in perm_b[:40]]
        held_out = [data_b[i] for i in perm_b[150:]]

        scores["zero-shot"].append(evaluate(pre, held_out)["f1"])

        ft = copy.deepcopy(pre)
        torch.manual_seed(seed)
        train_model(ft, leaked, epochs=30, lr=0.003, seed=seed)
        scores["fine-tuned"].append(evaluate(ft, held_out)["f1"])

        sc = TuningDiagnostic(QUERY_FEATURE_DIM, INDEX_FEATURE_DIM)
        torch.manual_seed(seed)
        train_model(sc, leaked, epochs=80, seed=seed)
        scores["from-scratch"].append(evaluate(sc, held_out)["f1"])

    for k, v in scores.items():
        print(f"  {k:12s}: f1 per seed={[round(x, 3) for x in v]}, "
              f"mean={np.mean(v):.3f}", flush=True)
    assert np.mean(scores["fine-tuned"]) > np.mean(scores["from-scratch"])
    assert np.mean(scores["fine-tuned"]) > np.mean(scores["zero-shot"])

    print("\nALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
