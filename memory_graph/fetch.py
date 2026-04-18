"""
Fetch utilities for MemPlan views.

Downstream algorithms (ComposeMatch, etc.) will call into this module to obtain
memory-backed planlets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .schema import MemoryGraph, MEdge, MNode


@dataclass
class MemorySubgraph:
    """Lightweight container for a matched planlet."""

    nodes: List[MNode]
    edges: List[MEdge]
    scope: Tuple[str, str]  # (city, date)


def planlet_embeddings(
    view_id: str, scopes: Sequence[Tuple[str, str]], graph: MemoryGraph
) -> Iterable[MemorySubgraph]:
    """
    Stub implementation: return action nodes that fall into the requested (city, date)
    scopes.  This is primarily for wiring tests; actual planlet mining will replace it.
    """

    for (city, date) in scopes:
        node_ids = graph.city_date_index.get((city, date), [])
        nodes = [graph.get_node(node_id) for node_id in node_ids]
        nodes = [node for node in nodes if node is not None]
        if not nodes:
            continue

        # Collect incident edges for inspection.
        relevant_ids = {node.id for node in nodes}
        edges = [
            edge
            for edge in graph.edges
            if edge.src in relevant_ids or edge.dst in relevant_ids
        ]

        yield MemorySubgraph(nodes=nodes, edges=edges, scope=(city, date))

