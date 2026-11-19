from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from src.collection import CollectionConfig, YouTubeCollector
from src.comparison import ComparisonResult, compare_analyses
from src.demo import demo_datasets
from src.graphs import build_graph, cumulative_snapshot, prune_visual_leaves
from src.metrics import NetworkAnalysis, analyze_network
from src.reporting import markdown_report
from src.visualization import network_html


st.set_page_config(
    page_title="YouTube Topology Comparator",
    page_icon="◉",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1500px; padding-top: 1.4rem}
      [data-testid="stMetric"] {border:1px solid #e5e7eb; border-radius:12px; padding:12px; background:#fafafa}
      .stAlert {border-radius:10px}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    st.session_state.setdefault("datasets", None)
    st.session_state.setdefault("root_ids", {})
    st.session_state.setdefault("collection_meta", {})


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or "network"


def export_bundle(
    datasets: dict[str, pd.DataFrame],
    analyses: dict[str, NetworkAnalysis],
    comparison: ComparisonResult,
    report: str,
    metadata: dict,
) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))
        archive.writestr("comparison/report.md", report)
        archive.writestr("comparison/raw_features.csv", comparison.raw_features.to_csv())
        archive.writestr("comparison/transformed_features.csv", comparison.transformed_features.to_csv())
        archive.writestr("comparison/normalized_features.csv", comparison.normalized_features.to_csv())
        archive.writestr("comparison/similarity_matrix.csv", comparison.similarity_matrix.to_csv())

        for name, edges in datasets.items():
            stem = safe_name(name)
            analysis = analyses[name]
            archive.writestr(f"data/{stem}_edges.csv", edges.to_csv(index=False))
            archive.writestr(f"metrics/{stem}_node_metrics.csv", analysis.node_metrics.to_csv(index=False))
            archive.writestr(f"metrics/{stem}_summary.json", json.dumps(analysis.features, indent=2))
            archive.writestr(f"visualizations/{stem}.html", network_html(analysis.graph))

            gexf = io.BytesIO()
            nx.write_gexf(analysis.graph, gexf)
            archive.writestr(f"networks/{stem}.gexf", gexf.getvalue())

    return payload.getvalue()


def analyze_all(
    datasets: dict[str, pd.DataFrame],
    root_ids: dict[str, str],
    normalization: str,
) -> tuple[dict[str, NetworkAnalysis], ComparisonResult]:
    analyses = {
        name: analyze_network(build_graph(edges), root_channel_id=root_ids.get(name))
        for name, edges in datasets.items()
    }
    return analyses, compare_analyses(analyses, normalization=normalization)


init_state()

st.title("YouTube Topology Comparator")
st.caption("Compare observable subscription networks using structural metrics and cosine similarity")

with st.sidebar:
    st.header("Configuration")
    source_mode = st.radio("Data source", ["Demonstration", "YouTube API", "CSV files"])
    depth = st.slider("Maximum collection depth", min_value=1, max_value=3, value=1)
    max_subscriptions = st.number_input(
        "Subscriptions per visited channel", min_value=10, max_value=500, value=100, step=10
    )
    normalization = st.selectbox(
        "Feature normalization",
        ["auto", "zscore", "maxabs", "none"],
        help="Auto uses z-score for 3+ networks and max-absolute scaling for exactly 2.",
    )
    st.divider()
    st.caption("Edge convention: subscriber → subscribed channel")

st.info(
    "YouTube exposes only public subscription lists. Missing outgoing edges may represent privacy settings, not an absence of subscriptions."
)

if source_mode == "Demonstration":
    st.write("Load six deterministic networks representing different structural profiles. No API key is required.")
    if st.button("Load demonstration", type="primary", width="stretch"):
        datasets, roots = demo_datasets()
        st.session_state.datasets = datasets
        st.session_state.root_ids = roots
        st.session_state.collection_meta = {
            "source": "demonstration",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

elif source_mode == "YouTube API":
    api_key = st.text_input("YouTube Data API key", type="password")
    channel_text = st.text_area(
        "Reference channels (one per line)",
        placeholder="@channelA\nhttps://www.youtube.com/@channelB",
        height=130,
    )
    if st.button("Collect and analyze", type="primary", width="stretch"):
        entries = [line.strip() for line in channel_text.splitlines() if line.strip()]
        if not api_key:
            st.error("Provide a YouTube Data API key.")
        elif len(entries) < 2:
            st.error("Provide at least two reference channels.")
        else:
            try:
                collector = YouTubeCollector(api_key)
                config = CollectionConfig(
                    max_depth=int(depth),
                    max_subscriptions_per_channel=int(max_subscriptions),
                )
                progress = st.progress(0.0, "Resolving channels")
                datasets: dict[str, pd.DataFrame] = {}
                roots: dict[str, str] = {}
                warnings: list[str] = []

                for index, entry in enumerate(entries):
                    channel = collector.resolve_channel(entry)
                    progress.progress(index / len(entries), f"Collecting {channel.title}")
                    result = collector.collect(channel, config)
                    datasets[channel.title] = result.edges
                    roots[channel.title] = channel.channel_id
                    warnings.extend(result.warnings)

                progress.progress(1.0, "Collection completed")
                st.session_state.datasets = datasets
                st.session_state.root_ids = roots
                st.session_state.collection_meta = {
                    "source": "youtube_api",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "max_depth": int(depth),
                    "max_subscriptions_per_channel": int(max_subscriptions),
                    "warnings": warnings,
                }
                if warnings:
                    st.warning(f"Collection completed with {len(warnings)} warning(s). See the exported metadata.")
            except Exception as exc:
                st.exception(exc)

else:
    st.write(
        "Upload one CSV edge list for each reference channel. Required columns: "
        "`source_channel_id` and `target_channel_id`."
    )
    uploads = st.file_uploader("Network edge lists", type="csv", accept_multiple_files=True)
    if st.button("Load and analyze", type="primary", width="stretch"):
        if len(uploads) < 2:
            st.error("Upload at least two CSV files.")
        else:
            datasets = {upload.name.rsplit(".", 1)[0]: pd.read_csv(upload) for upload in uploads}
            st.session_state.datasets = datasets
            st.session_state.root_ids = {}
            st.session_state.collection_meta = {
                "source": "csv_upload",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }


if st.session_state.datasets:
    try:
        analyses, comparison = analyze_all(
            st.session_state.datasets,
            st.session_state.root_ids,
            normalization,
        )
    except Exception as exc:
        st.exception(exc)
        st.stop()

    if comparison.warning:
        st.warning(comparison.warning)

    names = list(analyses)
    tabs = st.tabs(
        ["Overview", "Network explorer", "Structural features", "Similarity", "Report and export"]
    )

    with tabs[0]:
        st.subheader("Network overview")
        selected_overview = st.selectbox("Channel", names, key="overview_channel")
        analysis = analyses[selected_overview]
        cols = st.columns(5)
        cols[0].metric("Nodes", analysis.graph.number_of_nodes())
        cols[1].metric("Edges", analysis.graph.number_of_edges())
        cols[2].metric("Density", f"{analysis.features['density']:.4f}")
        cols[3].metric("Modularity", f"{analysis.features['modularity']:.4f}")
        cols[4].metric("PR centralization", f"{analysis.features['pagerank_centralization']:.4f}")

        st.markdown("#### Most central channels")
        st.dataframe(
            analysis.node_metrics.sort_values("pagerank", ascending=False).head(15),
            width="stretch",
            hide_index=True,
        )

    with tabs[1]:
        selected_graph = st.selectbox("Channel", names, key="graph_channel")
        edges = st.session_state.datasets[selected_graph]
        max_available_depth = (
            int(pd.to_numeric(edges["depth"], errors="coerce").fillna(1).max())
            if "depth" in edges.columns and not edges.empty
            else 1
        )
        graph_depth = st.slider("Snapshot depth", 0, max_available_depth, max_available_depth)
        prune = st.toggle("Prune leaves for visualization", value=True)
        graph = cumulative_snapshot(edges, graph_depth)
        if prune:
            graph = prune_visual_leaves(graph)
        st.caption(f"Displayed graph: {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        components.html(network_html(graph), height=700, scrolling=False)

    with tabs[2]:
        st.subheader("Nine-dimensional structural signatures")
        st.dataframe(comparison.raw_features, width="stretch")
        chart_data = comparison.normalized_features.reset_index(names="channel").melt(
            "channel", var_name="feature", value_name="value"
        )
        fig = px.bar(
            chart_data,
            x="feature",
            y="value",
            color="channel",
            barmode="group",
            title=f"Normalized structural features ({comparison.normalization_used})",
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, width="stretch")

    with tabs[3]:
        st.subheader("Pairwise structural similarity")
        scale = "RdBu" if comparison.normalization_used == "zscore" else "Blues"
        zmin = -1 if comparison.normalization_used == "zscore" else 0
        heatmap = px.imshow(
            comparison.similarity_matrix,
            text_auto=".3f",
            zmin=zmin,
            zmax=1,
            color_continuous_scale=scale,
            title="Cosine-similarity matrix",
        )
        st.plotly_chart(heatmap, width="stretch")

        if len(names) >= 3:
            distance = 1 - comparison.similarity_matrix
            st.caption("The table below contains cosine distances and can be used for clustering.")
            st.dataframe(distance, width="stretch")

    with tabs[4]:
        report = markdown_report(analyses, comparison, st.session_state.collection_meta)
        st.markdown(report)
        metadata = {
            **st.session_state.collection_meta,
            "normalization_requested": normalization,
            "normalization_used": comparison.normalization_used,
            "channels": names,
        }
        bundle = export_bundle(
            st.session_state.datasets,
            analyses,
            comparison,
            report,
            metadata,
        )
        st.download_button(
            "Download complete analysis",
            bundle,
            file_name="youtube_topology_analysis.zip",
            mime="application/zip",
            width="stretch",
        )

