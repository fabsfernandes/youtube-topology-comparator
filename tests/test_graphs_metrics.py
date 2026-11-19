import pandas as pd

from src.graphs import build_graph, cumulative_snapshot, prune_visual_leaves
from src.metrics import FEATURE_NAMES, analyze_network


def sample_edges():
    return pd.DataFrame(
        {
            "source_channel_id": ["a", "a", "b", "c"],
            "target_channel_id": ["b", "c", "c", "a"],
            "depth": [1, 1, 2, 2],
        }
    )


def test_graph_and_features():
    graph = build_graph(sample_edges())
    analysis = analyze_network(graph, root_channel_id="a")
    assert graph.number_of_nodes() == 3
    assert list(analysis.features) == FEATURE_NAMES
    assert 0 <= analysis.features["pagerank_centralization"] <= 1 + 1e-9
    assert analysis.node_metrics["is_root"].sum() == 1


def test_snapshot_and_pruning_do_not_mutate_original():
    graph = cumulative_snapshot(sample_edges(), 1)
    original_nodes = graph.number_of_nodes()
    pruned = prune_visual_leaves(graph)
    assert graph.number_of_nodes() == original_nodes
    assert pruned.number_of_nodes() <= original_nodes

