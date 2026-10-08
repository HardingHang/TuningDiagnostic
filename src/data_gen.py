"""Training-data generation by simulation (Section 4.2).

A real deployment collects labels by running a comprehensive tuning tool
(e.g. Extend) plus optimizer what-if calls (HypoPG).  For a minimal,
self-contained verification we replace both with a deterministic mock:

  * SyntheticDatabase        -- random schema (DDL) + table/column statistics
  * make_query               -- random query instance over the schema
  * cost()                   -- mock cost of a query/workload under a config
  * mock_tuner()             -- greedy budget-constrained index selection
  * generate_dataset()       -- <w_i, c_{i-1}, b, theta> samples with labels

The label rule follows the paper exactly:
    label = 1  iff  (cost(w, c_old) - cost(w, c_new)) / cost(w, c_old) >= theta
"""

from dataclasses import dataclass, field

import numpy as np

from .featurizer import GraphSample, featurize_index_config, featurize_query
from .schema_graph import schema_to_graph

INDEX_SCAN_FACTOR = 0.05  # cost multiplier when a useful index exists


# --------------------------------------------------------------------------
# Synthetic database
# --------------------------------------------------------------------------

@dataclass
class SyntheticDatabase:
    ddl: str
    reltuples: dict[str, float]
    avg_width: dict[str, float]          # "tbl.col" -> bytes
    columns: dict[str, list[str]]        # table -> column names
    primary_keys: dict[str, str]         # table -> pk column
    foreign_keys: dict[str, tuple[str, str]]  # "tbl.col" -> (ref_tbl, ref_col)
    graph: object = field(init=False)

    def __post_init__(self):
        self.graph = schema_to_graph(self.ddl)

    @property
    def raw_data_size(self) -> float:
        return sum(
            self.reltuples[t] * sum(self.avg_width[f"{t}.{c}"] for c in cols)
            for t, cols in self.columns.items()
        )


def generate_database(rng: np.random.Generator, prefix: str, n_tables: int) -> SyntheticDatabase:
    """Random schema: chained tables t_i with a FK to t_{i-1}, random columns."""
    ddl_parts: list[str] = []
    reltuples: dict[str, float] = {}
    avg_width: dict[str, float] = {}
    columns: dict[str, list[str]] = {}
    primary_keys: dict[str, str] = {}
    foreign_keys: dict[str, tuple[str, str]] = {}

    type_pool = [("INT", 4), ("BIGINT", 8), ("DECIMAL", 8), ("DATE", 4),
                 ("VARCHAR(20)", 20), ("TEXT", 32)]

    for i in range(n_tables):
        t = f"{prefix}_t{i}"
        cols = ["id INT"]
        pk = "id"
        col_names = ["id"]
        avg_width[f"{t}.id"] = 4
        if i > 0:
            ref = f"{prefix}_t{rng.integers(0, i)}"
            cols.append("ref_id INT")
            col_names.append("ref_id")
            avg_width[f"{t}.ref_id"] = 4
            foreign_keys[f"{t}.ref_id"] = (ref, "id")
        for j in range(rng.integers(2, 5)):
            ty, width = type_pool[rng.integers(0, len(type_pool))]
            name = f"c{j}"
            cols.append(f"{name} {ty}")
            col_names.append(name)
            avg_width[f"{t}.{name}"] = width
        ddl_parts.append(f"CREATE TABLE {t} ({', '.join(cols)}, PRIMARY KEY ({pk}));")
        for src, (rt, rc) in list(foreign_keys.items()):
            st, sc = src.split(".")
            if st == t:
                ddl_parts.append(
                    f"ALTER TABLE {t} ADD FOREIGN KEY ({sc}) REFERENCES {rt}({rc});"
                )
        reltuples[t] = float(rng.integers(10_000, 1_000_000))
        columns[t] = col_names
        primary_keys[t] = pk

    return SyntheticDatabase(
        "\n".join(ddl_parts), reltuples, avg_width, columns, primary_keys, foreign_keys
    )


# --------------------------------------------------------------------------
# Query generation
# --------------------------------------------------------------------------

@dataclass
class QueryInstance:
    sql: str
    tables: list[str]
    filter_cols: dict[str, list[str]]
    join_cols: dict[str, list[str]]
    rows_filtered: dict[str, float]


def make_query(rng: np.random.Generator, db: SyntheticDatabase) -> QueryInstance:
    tables = list(db.columns)
    anchor = tables[rng.integers(0, len(tables))]
    use_join = any(k.startswith(f"{anchor}.") for k in db.foreign_keys) and rng.random() < 0.5

    tables_used = [anchor]
    filter_cols: dict[str, list[str]] = {anchor: []}
    join_cols: dict[str, list[str]] = {}
    rows_filtered: dict[str, float] = {}
    join_sql = ""

    if use_join:
        fk_col, (ref_t, ref_c) = next(
            (k.split(".")[1], v) for k, v in db.foreign_keys.items()
            if k.startswith(f"{anchor}.")
        )
        tables_used.append(ref_t)
        join_cols = {anchor: [fk_col], ref_t: [ref_c]}
        filter_cols[ref_t] = []
        join_sql = f" JOIN {ref_t} ON {anchor}.{fk_col} = {ref_t}.{ref_c}"

    sel_cols, where_cols = [], []
    for t in tables_used:
        candidates = [c for c in db.columns[t] if c not in ("id", "ref_id")]
        rng.shuffle(candidates)
        for c in candidates[: rng.integers(1, min(3, len(candidates) + 1))]:
            filter_cols[t].append(c)
            where_cols.append(f"{t}.{c} = 1")
            rows_filtered[t] = rows_filtered.get(t, 1.0) * float(rng.uniform(0.02, 0.4))
        proj = [c for c in db.columns[t] if c not in filter_cols[t]]
        sel_cols.append(f"{t}.{proj[0]}" if proj else f"{t}.id")

    where_sql = f" WHERE {' AND '.join(where_cols)}" if where_cols else ""
    sql = f"SELECT {', '.join(sel_cols)} FROM {anchor}{join_sql}{where_sql}"
    if rng.random() < 0.5 and where_cols:
        sql += f" GROUP BY {', '.join(sel_cols)}"
    if rng.random() < 0.3:
        sql += f" ORDER BY {sel_cols[0]}"
    return QueryInstance(sql, tables_used, filter_cols, join_cols, rows_filtered)


# --------------------------------------------------------------------------
# Mock cost model + mock tuning tool
# --------------------------------------------------------------------------

def _query_cost(q: QueryInstance, config: set, db: SyntheticDatabase) -> float:
    terms = []
    for t in q.tables:
        base = db.reltuples[t] * q.rows_filtered.get(t, 1.0)
        useful_cols = set(q.filter_cols.get(t, [])) | set(q.join_cols.get(t, []))
        has_index = any(
            it == t and icols and icols[0] in useful_cols for it, icols in config
        )
        terms.append(base * INDEX_SCAN_FACTOR if has_index else base)
    cost = sum(terms)
    if len(terms) == 2:
        cost += 0.2 * min(terms)  # join overhead
    return cost


def workload_cost(queries: list[QueryInstance], weights: list[float], config: set,
                  db: SyntheticDatabase) -> float:
    return sum(_query_cost(q, config, db) * w for q, w in zip(queries, weights))


def index_size(indexes: set, db: SyntheticDatabase) -> float:
    return sum(
        db.reltuples[t] * sum(db.avg_width[f"{t}.{c}"] for c in cols)
        for t, cols in sorted(indexes)
    )


def mock_tuner(queries: list[QueryInstance], weights: list[float],
               db: SyntheticDatabase, budget_bytes: float) -> set:
    """Greedy gain/size index selection over width-1 and width-2 candidates."""
    candidates: set = set()
    for q in queries:
        for t in q.tables:
            cols = list(dict.fromkeys(q.filter_cols.get(t, []) + q.join_cols.get(t, [])))
            for c in cols:
                candidates.add((t, (c,)))
            for a, b in zip(cols, cols[1:]):
                candidates.add((t, (a, b)))

    chosen: set = set()
    current = workload_cost(queries, weights, chosen, db)
    while True:
        best, best_ratio = None, 0.0
        for cand in sorted(candidates - chosen):
            trial = chosen | {cand}
            if index_size(trial, db) > budget_bytes:
                continue
            gain = current - workload_cost(queries, weights, trial, db)
            size = index_size({cand}, db)
            if gain > 0 and gain / size > best_ratio:
                best, best_ratio = cand, gain / size
        if best is None:
            return chosen
        chosen.add(best)
        current = workload_cost(queries, weights, chosen, db)


# --------------------------------------------------------------------------
# Dataset generation
# --------------------------------------------------------------------------

@dataclass
class DiagnosticSample:
    queries: list[GraphSample]
    weights: list[float]
    config: GraphSample
    budget: float
    theta: float
    label: int


def generate_dataset(rng: np.random.Generator, db: SyntheticDatabase,
                     n_samples: int, n_templates: int = 10, m_queries: int = 4,
                     thetas=(0.05, 0.10, 0.15), budgets=(0.25, 0.5, 0.75)
                     ) -> list[DiagnosticSample]:
    """Simulate rounds of workload drift; label <w_i, c_{i-1}, b, theta> pairs.

    As in the TPC-H/TPC-DS setting of the paper, a workload is a mixture of a
    fixed pool of query templates with random frequencies.
    """
    templates = [make_query(rng, db) for _ in range(n_templates)]
    template_graphs = [featurize_query(db.graph, t.sql, t.rows_filtered) for t in templates]

    def fresh_workload():
        idx = rng.choice(n_templates, size=m_queries, replace=False).tolist()
        return idx, rng.uniform(0.8, 1.2, size=m_queries).tolist()

    samples: list[DiagnosticSample] = []
    tidx, weights = fresh_workload()
    b = budgets[rng.integers(0, len(budgets))]
    config = mock_tuner([templates[i] for i in tidx], weights, db, b * db.raw_data_size)

    for _ in range(n_samples):
        # workload drift: unchanged / partial drift / fresh workload
        roll = rng.random()
        if roll < 0.34:
            weights = (np.asarray(weights) * rng.uniform(0.9, 1.1, size=len(weights))).tolist()
        elif roll < 0.67:
            n_replace = int(rng.integers(1, m_queries))
            pool = [i for i in range(n_templates) if i not in tidx]
            repl = rng.choice(pool, size=n_replace, replace=False).tolist()
            pos = rng.choice(m_queries, size=n_replace, replace=False).tolist()
            for p, r in zip(pos, repl):
                tidx[p] = r
            weights = rng.uniform(0.8, 1.2, size=m_queries).tolist()
        else:
            tidx, weights = fresh_workload()

        queries = [templates[i] for i in tidx]
        b = budgets[rng.integers(0, len(budgets))]
        theta = thetas[rng.integers(0, len(thetas))]
        new_config = mock_tuner(queries, weights, db, b * db.raw_data_size)

        cost_old = workload_cost(queries, weights, config, db)
        cost_new = workload_cost(queries, weights, new_config, db)
        label = int((cost_old - cost_new) / cost_old >= theta)

        samples.append(DiagnosticSample(
            queries=[template_graphs[i] for i in tidx],
            weights=[w / sum(weights) for w in weights],
            config=featurize_index_config(db.graph, sorted(config), db.reltuples, db.avg_width),
            budget=float(b),
            theta=float(theta),
            label=label,
        ))
        config = new_config
    return samples
