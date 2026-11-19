from __future__ import annotations

import networkx as nx
import pandas as pd


REQUIRED_COLUMNS = {"source_channel_id", "target_channel_id"}


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    missing = REQUIRED_COLUMNS - set(edges.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if edges.empty:
        raise ValueError("The edge list is empty. The channel may hide its subscriptions.")

    graph = nx.DiGraph()
    records = edges.fillna("").to_dict("records")
    for row in records:
        source = str(row["source_channel_id"]).strip()
        target = str(row["target_channel_id"]).strip()
        if not source or not target or source == target:
            continue

        graph.add_node(
            source,
            title=str(row.get("source_channel_title", source)) or source,
        )
        graph.add_node(
            target,
            title=str(row.get("target_channel_title", target)) or target,
            description=str(row.get("target_description", "")),
            published_at=str(row.get("target_published_at", "")),
        )
        graph.add_edge(
            source,
            target,
            depth=int(row.get("depth", 1) or 1),
            collected_at=str(row.get("collected_at", "")),
        )

    if graph.number_of_nodes() == 0:
        raise ValueError("No valid relationships remained after validation.")
    return graph


def cumulative_snapshot(edges: pd.DataFrame, max_depth: int) -> nx.DiGraph:
    if "depth" not in edges.columns:
        return build_graph(edges)
    depths = pd.to_numeric(edges["depth"], errors="coerce").fillna(1)
    selected = edges.loc[depths <= max_depth]
    if selected.empty:
        graph = nx.DiGraph()
        if not edges.empty:
            root = str(edges.iloc[0]["source_channel_id"])
            title = str(edges.iloc[0].get("source_channel_title", root))
            graph.add_node(root, title=title)
        return graph
    return build_graph(selected)


def prune_visual_leaves(graph: nx.DiGraph) -> nx.DiGraph:
    pruned = graph.copy()
    leaves = [
        node
        for node in pruned
        if pruned.out_degree(node) == 0 and pruned.in_degree(node) <= 1
    ]
    pruned.remove_nodes_from(leaves)
    return pruned

