# Learned Tuning Diagnostic

Official implementation accompanying the paper:

> **Enhancing Online Index Tuning with a Learned Tuning Diagnostic**
> Haitian Hang, Jianling Sun. DEXA 2023.

## Overview

For online index tuning under dynamic workloads, launching a comprehensive
tuning tool on every triggering event is prohibitively expensive. This project
implements a **learned tuning diagnostic** for the Monitor-Diagnose-Tune
paradigm: a lightweight classifier that decides *whether* a tuning session
would recommend a configuration outperforming the current one by more than a
threshold θ, so that resource-intensive tuning sessions are only launched when
worthwhile.

Key components:

- **Schema2Graph** — converts the database schema into a directed graph whose
  nodes are tables/columns and whose labeled edges capture PK/FK/belongs-to
  relations and two-column indexes (plus self-connections).
- **Transferable representation** — queries and index configurations are
  embedded with relational graph convolutional networks (R-GCN) on the schema
  graph. Features carry no schema literals (no table/column names), which
  enables **cross-database learning** and incremental training on schema
  changes.
- **Prediction layer** — a fully-connected network on
  `concat(workload embedding, configuration embedding, memory budget, θ)`
  decides whether to launch a tuning session.
- **End-to-end training** — representation and prediction layers are trained
  jointly; training samples `<w_i, c_{i-1}, b, θ>` are labeled by whether
  `(cost(w, c_old) − cost(w, c_new)) / cost(w, c_old) ≥ θ`.

## Repository layout

```
src/
├── schema_graph.py   # DDL -> schema relation graph                        (Sec. 4.1.1)
├── featurizer.py     # query / index-configuration featurization           (Sec. 4.1.2-4.1.3)
├── model.py          # R-GCN representation layers + prediction layer      (Sec. 4.1)
├── data_gen.py       # training-data generation by simulation              (Sec. 4.2)
└── train.py          # end-to-end training + functional checks             (Sec. 4.2, Sec. 5)
```

## Requirements & usage

```bash
pip install -r requirements.txt   # torch, sqlglot, numpy (CPU is sufficient)
python -m src.train               # ~13 min on CPU
```

`python -m src.train` runs the pipeline end to end and checks:

1. schema-graph construction from DDL;
2. query / index-configuration featurization (shapes, schema-independence);
3. end-to-end training: loss decreases and the diagnostic clearly outperforms
   the degenerate always-tune baseline (F1 ≈ 0.86 vs ≈ 0.65 in the current
   run);
4. cross-database learning: the model trained on one database runs zero-shot
   on a different schema, and with only a few leaked samples from the new
   database, fine-tuning the pretrained model beats training from scratch
   (3-seed mean F1; cf. Sec. 5.3 of the paper).

## Simulation harness

To keep the repository self-contained, `data_gen.py` ships a lightweight
simulator that stands in for the production integrations used in the paper:

| Production component (paper)      | Self-contained substitute                          |
|-----------------------------------|----------------------------------------------------|
| Extend / (Anytime) DTA tuner      | greedy gain-per-size index selector (width ≤ 2)    |
| PostgreSQL + HypoPG what-if calls | deterministic analytical cost model                |
| TPC-H / TPC-DS workloads          | random schemas + template-based workload mixtures  |

The interfaces follow the real components, so the tuning tool, cost estimator,
and workload source can be swapped for PostgreSQL/HypoPG and benchmark or
production traces without touching the model code.

## Notes

- Index configurations consider indexes of width ≤ 2 (cf. Kossmann et al.,
  PVLDB 2020).
- Use mini-batch training (default 32). With batch size 1, Adam does not
  converge on class-balanced data — per-sample gradient noise is amplified
  into a random walk by the second-moment normalization.
- Threshold θ is an input feature of the model; the simulation generates
  datasets with a fixed θ = 15% (the setting used in the paper's end-to-end
  evaluation).
