"""Semantic catalog loader and models for DataMind.

Loads datasets defined in config/datasets.yaml, providing strongly-typed access
to metrics, dimensions, tables, join graphs, and unavailable concepts.
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml
from pydantic import BaseModel, Field


class DateDimensionConfig(BaseModel):
    column: str
    table: str
    allowed_grains: List[str] = Field(default_factory=lambda: ["year", "quarter", "month", "day"])


class TableConfig(BaseModel):
    description: str
    columns: Dict[str, str]


class JoinPathConfig(BaseModel):
    from_table: str
    to_table: str
    on: str

    @classmethod
    def model_validate(cls, obj: Any, *args, **kwargs):
        if isinstance(obj, dict):
            # PyYAML turns unquoted 'on:' into boolean True
            if True in obj and "on" not in obj:
                obj["on"] = obj[True]
        return super().model_validate(obj, *args, **kwargs)


class MetricConfig(BaseModel):
    description: str
    expression: str
    required_tables: List[str]


class DimensionConfig(BaseModel):
    description: str
    expression: str
    table: str
    allowed_values: Optional[List[str]] = None


class DatasetConfig(BaseModel):
    name: str
    description: str
    database: str
    schema_name: str = Field(alias="schema")
    date_dimension: Optional[DateDimensionConfig] = None
    tables: Dict[str, TableConfig]
    join_paths: List[JoinPathConfig] = Field(default_factory=list)
    metrics: Dict[str, MetricConfig] = Field(default_factory=dict)
    dimensions: Dict[str, DimensionConfig] = Field(default_factory=dict)
    unavailable_concepts: Dict[str, str] = Field(default_factory=dict)

    def get_metric(self, name: str) -> Optional[MetricConfig]:
        return self.metrics.get(name.lower())

    def get_dimension(self, name: str) -> Optional[DimensionConfig]:
        return self.dimensions.get(name.lower())

    def get_unavailable_concept_reason(self, concept: str) -> Optional[str]:
        concept_lower = concept.lower()
        for k, reason in self.unavailable_concepts.items():
            if k in concept_lower:
                return reason
        return None

    def resolve_joins(self, target_tables: Set[str]) -> List[JoinPathConfig]:
        """Finds minimal spanning set of joins to connect all target tables.

        Uses BFS over the undirected join graph defined in join_paths.
        """
        target_tables = {t.upper() for t in target_tables}
        if len(target_tables) <= 1:
            return []

        # Build adjacency graph
        adj: Dict[str, List[Tuple[str, JoinPathConfig]]] = {}
        for edge in self.join_paths:
            u = edge.from_table.upper()
            v = edge.to_table.upper()
            adj.setdefault(u, []).append((v, edge))
            adj.setdefault(v, []).append((u, edge))

        # Pick a root table. If LINEITEM is in target_tables, start with LINEITEM, else ORDERS, else first
        root = "LINEITEM" if "LINEITEM" in target_tables else ("ORDERS" if "ORDERS" in target_tables else next(iter(target_tables)))
        if root not in adj and len(target_tables) > 1:
            return []

        # Find paths from root to each other target table using BFS
        visited_nodes: Set[str] = {root}
        selected_edges: List[JoinPathConfig] = []
        selected_edges_signatures: Set[str] = set()

        for dest in target_tables:
            if dest == root:
                continue
            # BFS path from root to dest
            queue = deque([(root, [])])
            seen_in_bfs = {root}
            found_path = None

            while queue:
                curr, path = queue.popleft()
                if curr == dest:
                    found_path = path
                    break
                for nxt, edge in adj.get(curr, []):
                    if nxt not in seen_in_bfs:
                        seen_in_bfs.add(nxt)
                        queue.append((nxt, path + [(curr, nxt, edge)]))

            if found_path:
                for u, v, edge in found_path:
                    sig = f"{min(u, v)}-{max(u, v)}"
                    if sig not in selected_edges_signatures:
                        selected_edges_signatures.add(sig)
                        selected_edges.append(edge)

        return selected_edges


class Catalog:
    """Manages loading and querying semantic datasets."""

    def __init__(self, config_path: Optional[str] = None):
        if not config_path:
            config_path = str(Path(__file__).resolve().parent.parent / "config" / "datasets.yaml")
        self.config_path = config_path
        self.datasets: Dict[str, DatasetConfig] = {}
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Semantic catalog configuration not found at {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        datasets_raw = raw.get("datasets", {})
        for ds_id, ds_data in datasets_raw.items():
            self.datasets[ds_id.lower()] = DatasetConfig(**ds_data)

    def get_dataset(self, dataset_id: Optional[str] = None) -> DatasetConfig:
        target = (dataset_id or os.getenv("DATAMIND_DATASET", "tpch")).lower()
        if target not in self.datasets:
            raise KeyError(f"Dataset '{target}' not found in catalog. Available: {list(self.datasets.keys())}")
        return self.datasets[target]


# Global catalog singleton helper
_CATALOG_INSTANCE: Optional[Catalog] = None


def get_catalog() -> Catalog:
    global _CATALOG_INSTANCE
    if _CATALOG_INSTANCE is None:
        _CATALOG_INSTANCE = Catalog()
    return _CATALOG_INSTANCE
