"""Featurization of queries/workloads and index configurations on the schema graph.

Implements Sections 4.1.2 and 4.1.3 of the paper.  All features are
schema-independent (no table/column name literals), which is what makes the
representation transferable across databases.

Query/workload node features (padded to one layout for table & column nodes):
    [table_reference, percentage_rows_filtered,
     dtype one-hot (len(DTYPES)),
     in_select, in_functional_call, in_filter, in_join, in_groupby, in_orderby]

Index-configuration node features:
    [is_indexed (single-column index), reltuples (normalized), avg_width (normalized)]
plus Index-Left/Index-Right edges for two-column indexes.
"""

from dataclasses import dataclass, field

import numpy as np
import sqlglot
from sqlglot import exp

from .schema_graph import DTYPES, SchemaGraph

QUERY_FEATURE_DIM = 2 + len(DTYPES) + 6
INDEX_FEATURE_DIM = 3

# Offsets inside the query feature vector.
F_TABLE_REF = 0
F_ROWS_FILTERED = 1
F_DTYPE = 2
F_IN_SELECT = F_DTYPE + len(DTYPES)
F_IN_FUNC = F_IN_SELECT + 1
F_IN_FILTER = F_IN_FUNC + 1
F_IN_JOIN = F_IN_FILTER + 1
F_IN_GROUPBY = F_IN_JOIN + 1
F_IN_ORDERBY = F_IN_GROUPBY + 1


@dataclass
class GraphSample:
    """Featurized graph ready for the R-GCN."""

    features: np.ndarray                 # [num_nodes, feature_dim]
    edges: list[tuple[int, int, int]]    # (src, rel_id, dst)
    readout: list[int]                   # node ids used by sum pooling
    torch_cache: dict = field(default_factory=dict, repr=False, compare=False)


def _collect_columns(node) -> list[exp.Column]:
    if node is None:
        return []
    return list(node.find_all(exp.Column))


def featurize_query(
    graph: SchemaGraph,
    sql: str,
    rows_filtered: dict[str, float] | None = None,
) -> GraphSample:
    """Featurize one query against the schema graph.

    rows_filtered maps a table name to the percentage of rows kept by its
    filter conditions (from the DBMS optimizer in a real deployment; supplied
    by the caller here).  Tables absent from the map default to 1.0.
    """
    rows_filtered = rows_filtered or {}
    feats = np.zeros((len(graph.nodes), QUERY_FEATURE_DIM), dtype=np.float32)

    query = sqlglot.parse_one(sql, dialect="postgres")
    select = query if isinstance(query, exp.Select) else query.find(exp.Select)
    if select is None:
        raise ValueError(f"not a SELECT query: {sql}")

    referenced: set[int] = set()

    def mark(col: exp.Column, bit: int):
        table, name = col.table, col.name
        if not table or f"{table}.{name}" not in graph.name2id:
            return
        nid = graph.name2id[f"{table}.{name}"]
        feats[nid, bit] = 1.0
        referenced.add(nid)

    # --- table features -------------------------------------------------
    for tbl in select.find_all(exp.Table):
        name = tbl.name
        if name not in graph.name2id:
            continue
        nid = graph.name2id[name]
        feats[nid, F_TABLE_REF] = 1.0
        feats[nid, F_ROWS_FILTERED] = float(rows_filtered.get(name, 1.0))
        referenced.add(nid)

    # --- per-column data type + clause features --------------------------
    for col in _collect_columns(select):
        table = col.table
        if not table or f"{table}.{col.name}" not in graph.name2id:
            continue
        nid = graph.name2id[f"{table}.{col.name}"]
        feats[nid, F_DTYPE + DTYPES.index(graph.dtype[nid])] = 1.0
        referenced.add(nid)
        # any function-call ancestor -> in_functional_call
        node = col.parent
        while node is not None and node is not select:
            if isinstance(node, exp.Func):
                mark(col, F_IN_FUNC)
                break
            node = node.parent

    for expr in select.expressions:  # SELECT clause
        for col in _collect_columns(expr):
            mark(col, F_IN_SELECT)
    for col in _collect_columns(select.args.get("where")):
        mark(col, F_IN_FILTER)
    for join in select.args.get("joins", []):
        for col in _collect_columns(join.args.get("on")):
            mark(col, F_IN_JOIN)
    for col in _collect_columns(select.args.get("group")):
        mark(col, F_IN_GROUPBY)
    order = select.args.get("order")
    if order is None and isinstance(select.parent, exp.Order):
        order = select.parent
    for col in _collect_columns(order):
        mark(col, F_IN_ORDERBY)

    return GraphSample(feats, graph.edges, sorted(referenced))


def featurize_index_config(
    graph: SchemaGraph,
    indexes: list[tuple[str, tuple[str, ...]]],
    reltuples: dict[str, float],
    avg_width: dict[str, float],
) -> GraphSample:
    """Featurize an index configuration.

    indexes: list of (table, (col,) or (col1, col2)) -- width 1 or 2 only.
    reltuples: row count per table (normalized internally by the max).
    avg_width: average byte width per "tbl.col" (normalized internally by the max).
    """
    feats = np.zeros((len(graph.nodes), INDEX_FEATURE_DIM), dtype=np.float32)
    max_rows = max(reltuples.values(), default=1.0) or 1.0
    max_width = max(avg_width.values(), default=1.0) or 1.0

    edges = graph.edges
    readout: set[int] = set()
    for table, cols in indexes:
        if len(cols) == 1:
            nid = graph.name2id[f"{table}.{cols[0]}"]
            feats[nid, 0] = 1.0  # is_indexed
            readout.add(nid)
        elif len(cols) == 2:
            edges = graph.add_index_edges(table, cols[0], cols[1]).edges
            for c in cols:
                readout.add(graph.name2id[f"{table}.{c}"])
        else:
            raise ValueError("only index width 1 or 2 is supported")

    # Underlying data characteristics for every column node.
    for nid, name in enumerate(graph.nodes):
        if graph.is_table[nid]:
            continue
        table = graph.table_of(nid)
        feats[nid, 1] = reltuples.get(table, 0.0) / max_rows
        feats[nid, 2] = avg_width.get(name, 0.0) / max_width

    return GraphSample(feats, edges, sorted(readout))
