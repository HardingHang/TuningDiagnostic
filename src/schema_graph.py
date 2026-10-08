"""Schema2Graph: convert database schema (DDL) to a directed relation graph.

Implements Section 4.1.1 of the paper:
  - nodes: every table and every column
  - edges: 4 edge categories with labeled relations (see RELATIONS below),
    plus the two index edges used by the index-configuration featurization
    (Section 4.1.3) and a self-connection edge per node (Section 4.1.2).
"""

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

# Relation labels.  The first 9 capture the schema definition (Table 1 of the
# paper); INDEX_LEFT / INDEX_RIGHT mark two-column indexes (Section 4.1.3);
# SELF is the self-connection edge added before the R-GCN propagation.
FK_LEFT = "FK-Left"                # (col, col):  src is a foreign key for dst
FK_RIGHT = "FK-Right"              # (col, col):  dst is a foreign key for src
PK_LEFT = "PK-Left"                # (col, tbl):  src is a primary key of dst
BELONGS_LEFT = "Belongs-to-Left"   # (col, tbl):  src is a non-PK column of dst
PK_RIGHT = "PK-Right"              # (tbl, col):  dst is a primary key of src
BELONGS_RIGHT = "Belongs-to-Right" # (tbl, col):  dst is a non-PK column of src
FK_TBL_LEFT = "FK-Table-Left"      # (tbl, tbl):  src has a foreign key in dst
FK_TBL_RIGHT = "FK-Table-Right"    # (tbl, tbl):  dst has a foreign key in src
FK_TBL_BOTH = "FK-Table-Both"      # (tbl, tbl):  foreign keys in both directions
INDEX_LEFT = "Index-Left"          # (col, col):  two-column index (src, dst)
INDEX_RIGHT = "Index-Right"        # (col, col):  two-column index (dst, src)
SELF = "Self"                      # self-connection edge

RELATIONS = [
    FK_LEFT, FK_RIGHT,
    PK_LEFT, BELONGS_LEFT,
    PK_RIGHT, BELONGS_RIGHT,
    FK_TBL_LEFT, FK_TBL_RIGHT, FK_TBL_BOTH,
    INDEX_LEFT, INDEX_RIGHT,
    SELF,
]
REL2ID = {r: i for i, r in enumerate(RELATIONS)}
NUM_RELATIONS = len(RELATIONS)


@dataclass
class SchemaGraph:
    """Directed schema graph G_schema = (V, E, R)."""

    nodes: list[str]                      # node name: "tbl" or "tbl.col"
    is_table: list[bool]                  # True for table nodes
    dtype: list[str]                      # coarse data type per column ("" for tables)
    edges: list[tuple[int, int, int]]     # (src, rel_id, dst)
    name2id: dict[str, int] = field(init=False)

    def __post_init__(self):
        self.name2id = {n: i for i, n in enumerate(self.nodes)}

    def table_of(self, node_id: int) -> str:
        return self.nodes[node_id].split(".")[0]

    def add_index_edges(self, table: str, col_a: str, col_b: str) -> "SchemaGraph":
        """Return a copy with Index-Left/Index-Right edges for a two-column index."""
        a = self.name2id[f"{table}.{col_a}"]
        b = self.name2id[f"{table}.{col_b}"]
        edges = self.edges + [(a, REL2ID[INDEX_LEFT], b), (b, REL2ID[INDEX_RIGHT], a)]
        return SchemaGraph(self.nodes, self.is_table, self.dtype, edges)


def _coarse_type(type_sql: str) -> str:
    t = type_sql.upper()
    if any(k in t for k in ("INT", "SERIAL")):
        return "int"
    if any(k in t for k in ("DECIMAL", "NUMERIC", "REAL", "DOUBLE", "FLOAT")):
        return "float"
    if any(k in t for k in ("DATE", "TIME")):
        return "date"
    if "BOOL" in t:
        return "bool"
    if any(k in t for k in ("CHAR", "TEXT", "UUID", "VARCHAR")):
        return "text"
    return "other"


DTYPES = ["int", "float", "text", "date", "bool", "other"]


def schema_to_graph(ddl: str) -> SchemaGraph:
    """Parse CREATE TABLE / ALTER TABLE statements into a SchemaGraph."""
    nodes: list[str] = []
    is_table: list[bool] = []
    dtype: list[str] = []
    edges: list[tuple[int, int, int]] = []
    name2id: dict[str, int] = {}

    def node_id(name: str, tbl: bool, dt: str = "") -> int:
        if name not in name2id:
            name2id[name] = len(nodes)
            nodes.append(name)
            is_table.append(tbl)
            dtype.append(dt)
        return name2id[name]

    primary_keys: dict[str, set[str]] = {}
    foreign_keys: list[tuple[str, str, str, str]] = []  # (tbl, col, ref_tbl, ref_col)

    for stmt in sqlglot.parse(ddl, dialect="postgres"):
        if isinstance(stmt, exp.Create) and stmt.kind == "TABLE":
            schema = stmt.this
            table = schema.this.name
            node_id(table, True)
            pk_cols: set[str] = set()
            for expr in schema.expressions:
                if isinstance(expr, exp.ColumnDef):
                    col = expr.name
                    node_id(f"{table}.{col}", False, _coarse_type(expr.args["kind"].sql()))
                    for constraint in expr.args.get("constraints", []):
                        if isinstance(constraint, exp.ColumnConstraint):
                            if isinstance(constraint.kind, exp.PrimaryKeyColumnConstraint):
                                pk_cols.add(col)
                            elif isinstance(constraint.kind, exp.ReferenceColumnConstraint):
                                ref = constraint.kind.this
                                foreign_keys.append(
                                    (table, col, ref.this.name, ref.expressions[0].name)
                                )
                elif isinstance(expr, exp.PrimaryKey):
                    pk_cols.update(c.name for c in expr.expressions)
                elif isinstance(expr, exp.ForeignKey):
                    cols = [c.name for c in expr.expressions]
                    ref = expr.args["reference"]
                    ref_tbl = ref.this.this.name
                    ref_cols = [c.name for c in ref.this.expressions]
                    for c, rc in zip(cols, ref_cols):
                        foreign_keys.append((table, c, ref_tbl, rc))
            primary_keys[table] = pk_cols
        elif isinstance(stmt, exp.Alter):
            # ALTER TABLE t ADD PRIMARY KEY (c) / ADD FOREIGN KEY ... REFERENCES ...
            table = stmt.this.name
            for action in stmt.args.get("actions", []):
                if isinstance(action, exp.AddConstraint):
                    for constraint in action.expressions:
                        if isinstance(constraint, exp.PrimaryKey):
                            primary_keys.setdefault(table, set()).update(
                                c.name for c in constraint.expressions
                            )
                        elif isinstance(constraint, exp.ForeignKey):
                            cols = [c.name for c in constraint.expressions]
                            ref = constraint.args["reference"]
                            ref_tbl = ref.this.this.name
                            ref_cols = [c.name for c in ref.this.expressions]
                            for c, rc in zip(cols, ref_cols):
                                foreign_keys.append((table, c, ref_tbl, rc))

    # Column -> table edges (PK vs plain column).
    for name, tbl_flag in zip(nodes, is_table):
        if tbl_flag:
            continue
        table, col = name.split(".", 1)
        t_id = name2id[table]
        c_id = name2id[name]
        if col in primary_keys.get(table, set()):
            edges.append((c_id, REL2ID[PK_LEFT], t_id))
            edges.append((t_id, REL2ID[PK_RIGHT], c_id))
        else:
            edges.append((c_id, REL2ID[BELONGS_LEFT], t_id))
            edges.append((t_id, REL2ID[BELONGS_RIGHT], c_id))

    # Foreign-key edges, column-level and table-level.
    fk_tables: set[tuple[str, str]] = set()
    for table, col, ref_table, ref_col in foreign_keys:
        if ref_table not in name2id or f"{ref_table}.{ref_col}" not in name2id:
            continue
        c_id = node_id(f"{table}.{col}", False)
        rc_id = node_id(f"{ref_table}.{ref_col}", False)
        edges.append((c_id, REL2ID[FK_LEFT], rc_id))
        edges.append((rc_id, REL2ID[FK_RIGHT], c_id))
        fk_tables.add((table, ref_table))

    for table, ref_table in sorted(fk_tables):
        both = (ref_table, table) in fk_tables
        t_id, r_id = name2id[table], name2id[ref_table]
        if both:
            if table < ref_table:  # add the pair once
                edges.append((t_id, REL2ID[FK_TBL_BOTH], r_id))
                edges.append((r_id, REL2ID[FK_TBL_BOTH], t_id))
        else:
            edges.append((t_id, REL2ID[FK_TBL_LEFT], r_id))
            edges.append((r_id, REL2ID[FK_TBL_RIGHT], t_id))

    # Self-connection edges (Section 4.1.2).
    for i in range(len(nodes)):
        edges.append((i, REL2ID[SELF], i))

    return SchemaGraph(nodes, is_table, dtype, edges)
