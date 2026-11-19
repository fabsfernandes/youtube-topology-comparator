from __future__ import annotations

from src.comparison import ComparisonResult
from src.metrics import NetworkAnalysis


def markdown_report(
    analyses: dict[str, NetworkAnalysis],
    comparison: ComparisonResult,
    metadata: dict,
) -> str:
    names = list(analyses)
    pairs = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            pairs.append((float(comparison.similarity_matrix.loc[left, right]), left, right))
    pairs.sort(reverse=True)
    most = pairs[0]
    least = pairs[-1]

    lines = [
        "## Automated topology comparison report",
        "",
        f"The analysis contains **{len(names)} observable subscription networks**. "
        f"Structural signatures were compared with cosine similarity after "
        f"**{comparison.normalization_used} normalization**.",
        "",
        "### Network summary",
        "",
        "| Channel | Nodes | Edges | Density | Modularity | PageRank centralization |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, analysis in analyses.items():
        feature = analysis.features
        lines.append(
            f"| {name} | {int(feature['number_of_nodes'])} | {int(feature['number_of_edges'])} | "
            f"{feature['density']:.5f} | {feature['modularity']:.5f} | "
            f"{feature['pagerank_centralization']:.5f} |"
        )

    lines.extend(
        [
            "",
            "### Pairwise comparison",
            "",
            f"The most similar pair is **{most[1]} / {most[2]}** "
            f"(cosine similarity = **{most[0]:.4f}**).",
            "",
            f"The least similar pair is **{least[1]} / {least[2]}** "
            f"(cosine similarity = **{least[0]:.4f}**).",
            "",
            "These scores describe similarity in selected topological properties. They do not establish "
            "similarity of content, audience, musical genre, or political orientation, and they must be "
            "interpreted in light of unavailable private subscriptions.",
            "",
            "### Reproducibility",
            "",
            f"- Source: `{metadata.get('source', 'unknown')}`",
            f"- Collection timestamp: `{metadata.get('created_at', 'not recorded')}`",
            f"- Requested maximum depth: `{metadata.get('max_depth', 'not recorded')}`",
            f"- Subscription limit: `{metadata.get('max_subscriptions_per_channel', 'not recorded')}`",
        ]
    )
    return "\n".join(lines)

