"""
Persistence helpers for :mod:`memory_graph`.

We store each plan graph as JSONL so it can be streamed and inspected easily.
For interoperability, we also support emitting an RDF-style N-Triples file that
encodes edges and node attributes as triples.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape
from typing import Any, Dict, Iterable, Iterator, List, Optional, TextIO

from .schema import MEdge, MNode, PlanGraph


def serialise_plan_graph(graph: PlanGraph) -> Dict[str, Any]:
    """Convert a :class:`~memory_graph.schema.PlanGraph` into a JSON-serialisable dict."""

    return {
        "plan_id": graph.plan_id,
        "nodes": [{"id": node.id, "type": node.type, "attrs": dict(node.attrs)} for node in graph.nodes],
        "edges": [
            {"src": edge.src, "dst": edge.dst, "type": edge.type, "attrs": dict(edge.attrs)}
            for edge in graph.edges
        ],
    }


def save_jsonl(graphs: Iterable[PlanGraph], path: str | Path) -> None:
    """Persist plan graphs to ``path`` as JSONL (one plan graph per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for graph in graphs:
            fp.write(json.dumps(serialise_plan_graph(graph), ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> Iterator[PlanGraph]:
    """Stream plan graphs from a JSONL file produced by :func:`save_jsonl`."""
    path = Path(path)
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            yield deserialise_plan_graph(payload)


def deserialise_plan_graph(payload: Dict[str, Any]) -> PlanGraph:
    """Reconstruct a :class:`~memory_graph.schema.PlanGraph` from JSON data."""
    plan_id = str(payload.get("plan_id") or "UNKNOWN_PLAN")
    graph = PlanGraph(plan_id=plan_id)
    for node_data in payload.get("nodes", []):
        graph.add_node(MNode(id=node_data["id"], type=node_data["type"], attrs=dict(node_data.get("attrs") or {})))
    for edge_data in payload.get("edges", []):
        graph.add_edge(
            MEdge(
                src=edge_data["src"],
                dst=edge_data["dst"],
                type=edge_data["type"],
                attrs=dict(edge_data.get("attrs") or {}),
            )
        )
    return graph


def _to_rdf_literal(value: object) -> str:
    if value is None:
        return "\"\""
    if isinstance(value, bool):
        return "\"true\"" if value else "\"false\""
    if isinstance(value, (int, float)):
        return f"\"{value}\""
    if isinstance(value, (dict, list)):
        encoded = json.dumps(value, ensure_ascii=False)
        escaped = encoded.replace("\\", "\\\\").replace("\"", "\\\"")
        return f"\"{escaped}\""
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace("\"", "\\\"")
    return f"\"{escaped}\""


def _node_iri(plan_id: str, node_id: str) -> str:
    safe_plan = str(plan_id).replace(" ", "_")
    safe_node = str(node_id).replace(" ", "_")
    return f"<urn:memplan:{safe_plan}:{safe_node}>"


def _predicate_iri(kind: str, name: str) -> str:
    safe = str(name).replace(" ", "_")
    return f"<urn:memplan:{kind}:{safe}>"


def save_rdf_nt(graphs: Iterable[PlanGraph], path: str | Path) -> None:
    """
    Write an RDF-style N-Triples file encoding:
      * edges as (src, edge_type, dst)
      * node types as (node, nodeType, literal)
      * node attrs as (node, attr:<key>, literal)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for graph in graphs:
            _write_graph_nt(graph, fp)


def _write_graph_nt(graph: PlanGraph, fp: TextIO) -> None:
    plan_id = graph.plan_id
    for node in graph.nodes:
        s = _node_iri(plan_id, node.id)
        fp.write(f"{s} {_predicate_iri('node', 'type')} {_to_rdf_literal(node.type)} .\n")
        for key, value in (node.attrs or {}).items():
            fp.write(f"{s} {_predicate_iri('attr', key)} {_to_rdf_literal(value)} .\n")
    for edge in graph.edges:
        s = _node_iri(plan_id, edge.src)
        o = _node_iri(plan_id, edge.dst)
        fp.write(f"{s} {_predicate_iri('edge', edge.type)} {o} .\n")
        for key, value in (edge.attrs or {}).items():
            fp.write(f"{s} {_predicate_iri('edgeAttr', f'{edge.type}:{key}')} {_to_rdf_literal(value)} .\n")


def _normalise_value(value: object) -> tuple[str, str]:
    """
    Convert a Python value into a (type, string) pair for value-nodes.

    The returned strings are stable and used for node identity (hashing), so
    changes here affect merge behaviour in GraphML exports.
    """

    if value is None:
        return "null", ""
    if isinstance(value, bool):
        return "bool", "true" if value else "false"
    if isinstance(value, int):
        return "int", str(value)
    if isinstance(value, float):
        return "float", repr(value)
    if isinstance(value, (dict, list)):
        return "json", json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "str", str(value)


def _iter_filled_leaves(key: str, value: object) -> Iterator[tuple[str, object]]:
    if value is None:
        return
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            path = f"{key}.{sub_key}" if key else str(sub_key)
            yield from _iter_filled_leaves(path, sub_value)
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_filled_leaves(key, item)
        return
    if value == "":
        return
    yield key, value


def save_graphml(graphs: Iterable[PlanGraph], path: str | Path) -> None:
    """
    Export a merged, connected GraphML view of multiple plan graphs.

    Design:
      - Plan/Phase/Action nodes are preserved (with their original ids).
      - Each *filled* attribute/param value is represented as a shared Value node.
      - Nodes from different plans become connected by sharing the same Value nodes.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    node_data: Dict[str, Dict[str, str]] = {}
    edge_set: set[tuple[str, str, str, str, str]] = set()
    edges: List[tuple[str, str, str, str, str]] = []

    def _add_node(node_id: str, data: Dict[str, str]) -> None:
        if node_id in node_data:
            return
        node_data[node_id] = data

    def _label_for_entity(node: MNode) -> str:
        if node.type == "Plan":
            return f"Plan {node.id}"
        if node.type == "Phase":
            attrs = node.attrs or {}
            phase_key = attrs.get("phase_key") or node.id
            phase_type = attrs.get("phase_type") or "Phase"
            return f"{phase_type}:{phase_key}"
        if node.type == "Action":
            attrs = node.attrs or {}
            return str(attrs.get("action_type") or "Action")
        return node.id

    def _value_node_id(value: object) -> str:
        value_type, value_text = _normalise_value(value)
        signature = f"{value_type}:{value_text}"
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()
        node_id = f"val::{value_type}:{digest}"
        if node_id not in node_data:
            label = value_text if len(value_text) <= 80 else value_text[:77] + "..."
            _add_node(
                node_id,
                {
                    "node_type": "Value",
                    "label": label,
                    "value": value_text,
                    "value_type": value_type,
                },
            )
        return node_id

    def _add_edge(src: str, dst: str, edge_type: str, *, key: str = "", attrs: Optional[Dict[str, object]] = None) -> None:
        attrs_text = ""
        if attrs:
            attrs_text = json.dumps(attrs, ensure_ascii=False, sort_keys=True)
        record = (src, dst, edge_type, key, attrs_text)
        if record in edge_set:
            return
        edge_set.add(record)
        edges.append(record)

    for graph in graphs:
        for node in graph.nodes:
            _add_node(
                node.id,
                {
                    "node_type": node.type,
                    "label": _label_for_entity(node),
                },
            )

        for edge in graph.edges:
            _add_edge(edge.src, edge.dst, edge.type, attrs=dict(edge.attrs or {}))

        for node in graph.nodes:
            attrs = dict(node.attrs or {})
            params = attrs.get("params")
            if isinstance(params, dict):
                for param_key, param_value in params.items():
                    for key_path, leaf in _iter_filled_leaves(str(param_key), param_value):
                        value_node = _value_node_id(leaf)
                        _add_edge(node.id, value_node, "hasParam", key=key_path)

            for attr_key, attr_value in attrs.items():
                if attr_key == "params":
                    continue
                for key_path, leaf in _iter_filled_leaves(str(attr_key), attr_value):
                    value_node = _value_node_id(leaf)
                    _add_edge(node.id, value_node, "hasAttr", key=key_path)

    def _xml(text: str) -> str:
        return escape(text, {'"': "&quot;"})

    with path.open("w", encoding="utf-8") as fp:
        fp.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fp.write(
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns '
            'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n'
        )
        fp.write('  <key id="n_type" for="node" attr.name="node_type" attr.type="string"/>\n')
        fp.write('  <key id="n_label" for="node" attr.name="label" attr.type="string"/>\n')
        fp.write('  <key id="n_value" for="node" attr.name="value" attr.type="string"/>\n')
        fp.write('  <key id="n_vtype" for="node" attr.name="value_type" attr.type="string"/>\n')
        fp.write('  <key id="e_type" for="edge" attr.name="edge_type" attr.type="string"/>\n')
        fp.write('  <key id="e_key" for="edge" attr.name="key" attr.type="string"/>\n')
        fp.write('  <key id="e_attrs" for="edge" attr.name="attrs" attr.type="string"/>\n')
        fp.write('  <graph id="G" edgedefault="directed">\n')

        for node_id in sorted(node_data):
            data = node_data[node_id]
            fp.write(f'    <node id="{_xml(node_id)}">\n')
            fp.write(f'      <data key="n_type">{_xml(data.get("node_type", ""))}</data>\n')
            fp.write(f'      <data key="n_label">{_xml(data.get("label", ""))}</data>\n')
            if data.get("node_type") == "Value":
                fp.write(f'      <data key="n_value">{_xml(data.get("value", ""))}</data>\n')
                fp.write(f'      <data key="n_vtype">{_xml(data.get("value_type", ""))}</data>\n')
            fp.write("    </node>\n")

        for idx, (src, dst, edge_type, key, attrs_text) in enumerate(edges):
            fp.write(f'    <edge id="e{idx}" source="{_xml(src)}" target="{_xml(dst)}">\n')
            fp.write(f'      <data key="e_type">{_xml(edge_type)}</data>\n')
            if key:
                fp.write(f'      <data key="e_key">{_xml(key)}</data>\n')
            if attrs_text:
                fp.write(f'      <data key="e_attrs">{_xml(attrs_text)}</data>\n')
            fp.write("    </edge>\n")

        fp.write("  </graph>\n")
        fp.write("</graphml>\n")


def serialise(graph: PlanGraph) -> Dict[str, Any]:
    """Backward-compatible wrapper for older code paths."""
    return serialise_plan_graph(graph)


def save_json(graph: PlanGraph, path: str | Path) -> None:
    """Backward-compatible single-graph JSON writer (non-JSONL)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(serialise_plan_graph(graph), fp, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> PlanGraph:
    """Backward-compatible single-graph JSON reader."""
    path = Path(path)
    with path.open(encoding="utf-8") as fp:
        payload = json.load(fp)
    return deserialise_plan_graph(payload)
