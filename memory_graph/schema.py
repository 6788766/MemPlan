"""
Core data structures for the MemPlan plan graph.

We model one plan instance with a task-agnostic backbone:

    Plan → Phase → Action

Optionally, a Phase can contain Subsection nodes (disabled by default in the
extractor config).

All task-specific semantics (e.g., "Move", "Eat", "Refactor") live in
``Action.attrs['action_type']`` and ``Action.attrs['params']`` and are driven by
config, not hard-coded in the schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional

NodeType = Literal["Plan", "Phase", "Action", "Subsection"]
EdgeType = Literal["hasPhase", "hasAction", "hasSubsection", "before"]


@dataclass(frozen=True)
class MNode:
    """Typed node in the plan graph."""

    id: str
    type: NodeType
    attrs: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MEdge:
    """Typed, directed edge between two nodes."""

    src: str
    dst: str
    type: EdgeType
    attrs: Dict[str, object] = field(default_factory=dict)


class PlanGraph:
    """
    Lightweight in-memory container for a single plan instance.

    The graph keeps node/edge collections only. Any task-specific indexing
    implemented outside the core schema (e.g., in task adapters).
    """

    def __init__(self, plan_id: str) -> None:
        self.plan_id = plan_id
        self._nodes: Dict[str, MNode] = {}
        self._edges: List[MEdge] = []

    # ------------------------------------------------------------------
    # Node/edge management
    # ------------------------------------------------------------------
    def add_node(self, node: MNode) -> None:
        if node.id in self._nodes:
            raise ValueError(f"Node id {node.id} already present.")
        self._nodes[node.id] = node

    def add_edge(self, edge: MEdge) -> None:
        if edge.src not in self._nodes or edge.dst not in self._nodes:
            raise ValueError(f"Edge {edge} references unknown nodes.")
        self._edges.append(edge)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def nodes(self) -> Iterable[MNode]:
        return self._nodes.values()

    @property
    def edges(self) -> Iterable[MEdge]:
        return list(self._edges)

    def get_node(self, node_id: str) -> Optional[MNode]:
        return self._nodes.get(node_id)

    def neighbours(self, node_id: str) -> List[MEdge]:
        return [edge for edge in self._edges if edge.src == node_id or edge.dst == node_id]

    def summary(self) -> Dict[str, object]:
        """
        Compact dictionary summarising node counts per type.
        Useful for quick sanity checks in scripts.
        """
        type_counter: Dict[str, int] = {}
        for node in self._nodes.values():
            type_counter[node.type] = type_counter.get(node.type, 0) + 1

        return {
            "plan_id": self.plan_id,
            "nodes_total": len(self._nodes),
            "edges_total": len(self._edges),
            "nodes_by_type": type_counter,
        }


# Backwards-compatible alias (older code imported MemoryGraph).
MemoryGraph = PlanGraph
