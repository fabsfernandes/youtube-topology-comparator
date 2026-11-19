from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd


FEATURE_NAMES = [
    "number_of_nodes",
    "number_of_edges",
    "density",
    "average_clustering",
    "modularity",
    "mean_degree_centrality",
    "mean_closeness_centrality",
    "mean_betweenness_centrality",
    "pagerank_centralization",
]


@dataclass
class NetworkAnalysis:
    graph: nx.DiGraph
    node_metrics: pd.DataFrame
    features: dict[str, float]
    communities: dict[str, int]


def _communities_and_modularity(graph: nx.DiGraph) -> tuple[dict[str, int], float]:
    undirected = graph.to_undirected()
    if undirected.number_of_edges() == 0:
        return {node: 0 for node in graph}, 0.0

    groups = list(nx.community.greedy_modularity_communities(undirected))
    mapping = {node: index for index, group in enumerate(groups) for node in group}
    modularity = nx.community.modularity(undirected, groups)
    return mapping, float(modularity)


def _pagerank_centralization(graph: nx.DiGraph, pagerank: dict[str, float], alpha: float) -> float:
    n = graph.number_of_nodes()
    if n <= 2:
        return 0.0

    maximum = max(pagerank.values())
    numerator = sum(maximum - value for value in pagerank.values())

    # Freeman-style normalization using a directed in-star with the same n.
    star = nx.DiGraph()
    star.add_nodes_from(range(n))
    star.add_edges_from((leaf, 0) for leaf in range(1, n))
    ideal_rank = nx.pagerank(star, alpha=alpha)
    ideal_maximum = max(ideal_rank.values())
    denominator = sum(ideal_maximum - value for value in ideal_rank.values())
    return float(numerator / denominator) if denominator > 0 else 0.0


def analyze_network(
    graph: nx.DiGraph,
    root_channel_id: str | None = None,
    pagerank_alpha: float = 0.85,
) -> NetworkAnalysis:
    n = graph.number_of_nodes()
    if n == 0:
        raise ValueError("Cannot analyze an empty graph.")

    degree = nx.degree_centrality(graph)
    in_degree = nx.in_degree_centrality(graph)
    out_degree = nx.out_degree_centrality(graph)
    closeness = nx.closeness_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    clustering = nx.clustering(graph)
    pagerank = nx.pagerank(graph, alpha=pagerank_alpha)
    communities, modularity = _communities_and_modularity(graph)

    rows = []
    for node in graph:
        rows.append(
            {
                "channel_id": node,
                "channel_title": graph.nodes[node].get("title", node),
                "is_root": node == root_channel_id,
                "community": communities.get(node, 0),
                "in_degree": graph.in_degree(node),
                "out_degree": graph.out_degree(node),
                "degree_centrality": degree[node],
                "in_degree_centrality": in_degree[node],
                "out_degree_centrality": out_degree[node],
                "closeness_centrality": closeness[node],
                "betweenness_centrality": betweenness[node],
                "clustering_coefficient": clustering[node],
                "pagerank": pagerank[node],
            }
        )
    node_metrics = pd.DataFrame(rows)

    features = {
        "number_of_nodes": float(n),
        "number_of_edges": float(graph.number_of_edges()),
        "density": float(nx.density(graph)),
        "average_clustering": float(node_metrics["clustering_coefficient"].mean()),
        "modularity": modularity,
        "mean_degree_centrality": float(node_metrics["degree_centrality"].mean()),
        "mean_closeness_centrality": float(node_metrics["closeness_centrality"].mean()),
        "mean_betweenness_centrality": float(node_metrics["betweenness_centrality"].mean()),
        "pagerank_centralization": _pagerank_centralization(graph, pagerank, pagerank_alpha),
    }
    return NetworkAnalysis(graph, node_metrics, features, communities)

