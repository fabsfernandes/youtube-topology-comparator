from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


@dataclass(frozen=True)
class Channel:
    channel_id: str
    title: str
    description: str = ""
    published_at: str = ""


@dataclass(frozen=True)
class CollectionConfig:
    max_depth: int = 1
    max_subscriptions_per_channel: int = 100


@dataclass
class CollectionResult:
    edges: pd.DataFrame
    warnings: list[str]


EDGE_COLUMNS = [
    "source_channel_id",
    "source_channel_title",
    "target_channel_id",
    "target_channel_title",
    "target_description",
    "target_published_at",
    "depth",
    "collected_at",
]


class YouTubeCollector:
    def __init__(self, api_key: str):
        if not api_key.strip():
            raise ValueError("A YouTube Data API key is required.")
        self.api = build("youtube", "v3", developerKey=api_key.strip(), cache_discovery=False)

    def resolve_channel(self, value: str) -> Channel:
        raw = value.strip().rstrip("/")
        channel_id_match = re.search(r"(?:youtube\.com/channel/)?(UC[\w-]{20,})", raw)

        if channel_id_match:
            response = self.api.channels().list(
                part="snippet", id=channel_id_match.group(1)
            ).execute()
        else:
            handle_match = re.search(r"@([\w.-]+)", raw)
            handle = handle_match.group(1) if handle_match else raw
            response = self.api.channels().list(part="snippet", forHandle=handle).execute()

        items = response.get("items", [])
        if not items:
            raise ValueError(f"Channel not found: {value}")
        item = items[0]
        snippet = item.get("snippet", {})
        return Channel(
            channel_id=item["id"],
            title=snippet.get("title", item["id"]),
            description=snippet.get("description", ""),
            published_at=snippet.get("publishedAt", ""),
        )

    @staticmethod
    def _http_reason(exc: HttpError) -> str:
        try:
            body = json.loads(exc.content.decode("utf-8"))
            return body["error"]["errors"][0].get("reason", "unknown")
        except Exception:
            return "unknown"

    def _subscriptions(self, channel_id: str, limit: int) -> tuple[list[dict], str | None]:
        items: list[dict] = []
        token: str | None = None
        try:
            while len(items) < limit:
                response = self.api.subscriptions().list(
                    part="snippet",
                    channelId=channel_id,
                    maxResults=min(50, limit - len(items)),
                    pageToken=token,
                ).execute()
                items.extend(response.get("items", []))
                token = response.get("nextPageToken")
                if not token:
                    break
        except HttpError as exc:
            reason = self._http_reason(exc)
            if reason in {
                "subscriptionForbidden",
                "subscriberNotFound",
                "accountClosed",
                "accountSuspended",
                "forbidden",
            }:
                return [], reason
            raise
        return items, None

    def _channel_details(self, channel_ids: set[str]) -> dict[str, Channel]:
        details: dict[str, Channel] = {}
        ordered = sorted(channel_ids)
        for start in range(0, len(ordered), 50):
            response = self.api.channels().list(
                part="snippet", id=",".join(ordered[start : start + 50])
            ).execute()
            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                details[item["id"]] = Channel(
                    channel_id=item["id"],
                    title=snippet.get("title", item["id"]),
                    description=snippet.get("description", ""),
                    published_at=snippet.get("publishedAt", ""),
                )
        return details

    def collect(self, root: Channel, config: CollectionConfig) -> CollectionResult:
        if config.max_depth < 1:
            raise ValueError("Maximum depth must be at least 1.")

        queue = deque([(root.channel_id, root.title, 0)])
        visited: set[str] = set()
        records: list[dict] = []
        warnings: list[str] = []
        collected_at = pd.Timestamp.now(tz="UTC").isoformat()

        while queue:
            source_id, source_title, source_depth = queue.popleft()
            if source_id in visited or source_depth >= config.max_depth:
                continue
            visited.add(source_id)

            subscriptions, reason = self._subscriptions(
                source_id, config.max_subscriptions_per_channel
            )
            if reason:
                warnings.append(f"Could not expand {source_id}: {reason}")
                continue

            for item in subscriptions:
                snippet = item.get("snippet", {})
                target_id = snippet.get("resourceId", {}).get("channelId")
                if not target_id:
                    continue
                target_title = snippet.get("title", target_id)
                records.append(
                    {
                        "source_channel_id": source_id,
                        "source_channel_title": source_title,
                        "target_channel_id": target_id,
                        "target_channel_title": target_title,
                        "target_description": snippet.get("description", ""),
                        "target_published_at": "",
                        "depth": source_depth + 1,
                        "collected_at": collected_at,
                    }
                )
                if target_id not in visited:
                    queue.append((target_id, target_title, source_depth + 1))

        details = self._channel_details({r["target_channel_id"] for r in records}) if records else {}
        for record in records:
            detail = details.get(record["target_channel_id"])
            if detail:
                record["target_channel_title"] = detail.title
                record["target_description"] = detail.description
                record["target_published_at"] = detail.published_at

        edges = pd.DataFrame.from_records(records, columns=EDGE_COLUMNS)
        if not edges.empty:
            edges = edges.drop_duplicates(["source_channel_id", "target_channel_id"]).reset_index(drop=True)
        return CollectionResult(edges=edges, warnings=warnings)

