import numpy as np

from src.comparison import compare_analyses
from src.demo import demo_datasets
from src.graphs import build_graph
from src.metrics import analyze_network


def analyses():
    datasets, roots = demo_datasets()
    return {
        name: analyze_network(build_graph(edges), roots[name])
        for name, edges in datasets.items()
    }


def test_zscore_comparison_is_symmetric():
    result = compare_analyses(analyses(), normalization="zscore")
    assert result.similarity_matrix.shape == (6, 6)
    assert np.allclose(result.similarity_matrix, result.similarity_matrix.T)
    assert np.allclose(np.diag(result.similarity_matrix), 1)


def test_two_network_auto_uses_maxabs():
    subset = dict(list(analyses().items())[:2])
    result = compare_analyses(subset, normalization="auto")
    assert result.normalization_used == "maxabs"
    assert 0 <= result.similarity_matrix.iloc[0, 1] <= 1

