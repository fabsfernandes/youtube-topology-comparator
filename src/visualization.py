from __future__ import annotations

import networkx as nx
from pyvis.network import Network


PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5"]


def network_html(graph: nx.DiGraph) -> str:
    network = Network(
        height="660px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#111827",
        cdn_resources="in_line",
    )
    if graph.number_of_nodes() == 0:
        return network.generate_html()

    pagerank = nx.pagerank(graph) if graph.number_of_edges() else {node: 1.0 for node in graph}
    if graph.number_of_edges():
        groups = list(nx.community.greedy_modularity_communities(graph.to_undirected()))
        community = {node: index for index, group in enumerate(groups) for node in group}
    else:
        community = {node: 0 for node in graph}

    max_rank = max(pagerank.values()) or 1.0
    for node, attributes in graph.nodes(data=True):
        title = attributes.get("title", node)
        group = community.get(node, 0)
        network.add_node(
            node,
            label=title,
            title=f"{title}<br>PageRank: {pagerank.get(node, 0):.5f}<br>Community: {group}",
            size=12 + 34 * pagerank.get(node, 0) / max_rank,
            color=PALETTE[group % len(PALETTE)],
        )

    for source, target, attributes in graph.edges(data=True):
        network.add_edge(
            source,
            target,
            title=f"Depth: {attributes.get('depth', '')}",
            color="#9ca3af",
        )

    network.set_options(
        """
        {
          "physics": {
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {"gravitationalConstant": -55, "springLength": 105},
            "stabilization": {"iterations": 250}
          },
          "interaction": {"hover": true, "navigationButtons": true, "keyboard": true},
          "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.6}}}
        }
        """
    )
    return network.generate_html()

