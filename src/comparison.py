from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from src.metrics import FEATURE_NAMES, NetworkAnalysis


@dataclass
class ComparisonResult:
    raw_features: pd.DataFrame
    transformed_features: pd.DataFrame
    normalized_features: pd.DataFrame
    similarity_matrix: pd.DataFrame
    normalization_used: str
    warning: str | None = None


def compare_analyses(
    analyses: dict[str, NetworkAnalysis],
    normalization: str = "auto",
) -> ComparisonResult:
    if len(analyses) < 2:
        raise ValueError("At least two networks are required for comparison.")

    raw = pd.DataFrame(
        {name: [analysis.features[key] for key in FEATURE_NAMES] for name, analysis in analyses.items()},
        index=FEATURE_NAMES,
    ).T
    transformed = raw.copy()
    transformed["number_of_nodes"] = np.log1p(transformed["number_of_nodes"])
    transformed["number_of_edges"] = np.log1p(transformed["number_of_edges"])

    warning = None
    requested = normalization.lower()
    if requested == "auto":
        used = "zscore" if len(analyses) >= 3 else "maxabs"
    else:
        used = requested

    if used == "zscore" and len(analyses) == 2:
        warning = (
            "Z-score normalization with only two networks is degenerate: each varying feature becomes "
            "a symmetric pair. Max-absolute scaling is recommended for a direct two-channel comparison."
        )

    nonconstant = transformed.columns[transformed.nunique(dropna=False) > 1]
    if len(nonconstant) == 0:
        raise ValueError("All structural features are constant; similarity is undefined.")
    transformed = transformed.loc[:, nonconstant]

    if used == "zscore":
        values = StandardScaler().fit_transform(transformed)
    elif used == "maxabs":
        values = MaxAbsScaler().fit_transform(transformed)
    elif used == "none":
        values = transformed.to_numpy(dtype=float)
    else:
        raise ValueError("Normalization must be one of: auto, zscore, maxabs, none.")

    normalized = pd.DataFrame(values, index=transformed.index, columns=transformed.columns)
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0):
        raise ValueError(
            "At least one normalized feature vector has zero norm. Use maxabs normalization or add networks."
        )
    similarities = cosine_similarity(values)
    matrix = pd.DataFrame(similarities, index=raw.index, columns=raw.index)
    return ComparisonResult(raw, transformed, normalized, matrix, used, warning)

