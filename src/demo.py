from __future__ import annotations

import pandas as pd


def _frame(edges: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_channel_id": source,
                "source_channel_title": source,
                "target_channel_id": target,
                "target_channel_title": target,
                "target_description": "Demonstration channel",
                "target_published_at": "2020-01-01T00:00:00Z",
                "depth": depth,
                "collected_at": "2026-01-01T00:00:00Z",
            }
            for source, target, depth in edges
        ]
    )


def demo_datasets() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    datasets = {
        "Music Alpha": _frame([
            ("MA", "Label", 1), ("MA", "Artist1", 1), ("MA", "Artist2", 1),
            ("Artist1", "Label", 2), ("Artist2", "Label", 2), ("Label", "Media", 2),
            ("Media", "Artist1", 2),
        ]),
        "Music Beta": _frame([
            ("MB", "LabelB", 1), ("MB", "Singer1", 1), ("MB", "Singer2", 1),
            ("Singer1", "LabelB", 2), ("Singer2", "LabelB", 2), ("LabelB", "Press", 2),
            ("Press", "Singer2", 2),
        ]),
        "Music Gamma": _frame([
            ("MG", "Indie1", 1), ("MG", "Indie2", 1), ("MG", "Indie3", 1),
            ("Indie1", "Indie2", 2), ("Indie3", "MG", 2),
        ]),
        "News Alpha": _frame([
            ("NA", "Network", 1), ("NA", "Local1", 1), ("NA", "Local2", 1),
            ("Local1", "Network", 2), ("Local2", "Network", 2), ("Network", "Politics", 2),
            ("Politics", "NA", 2),
        ]),
        "News Beta": _frame([
            ("NB", "NetworkB", 1), ("NB", "Regional1", 1), ("NB", "Regional2", 1),
            ("Regional1", "NetworkB", 2), ("Regional2", "NetworkB", 2),
            ("NetworkB", "CurrentAffairs", 2), ("CurrentAffairs", "NB", 2),
        ]),
        "Digital News": _frame([
            ("DN", "Tech", 1), ("DN", "Creator", 1), ("DN", "Science", 1),
            ("Tech", "Science", 2), ("Creator", "DN", 2), ("Science", "Creator", 2),
            ("Tech", "Creator", 2),
        ]),
    }
    roots = {name: str(frame.iloc[0]["source_channel_id"]) for name, frame in datasets.items()}
    return datasets, roots

