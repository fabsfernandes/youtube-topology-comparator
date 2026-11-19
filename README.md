# YouTube Topology Comparator

A fully Python/Streamlit system for comparing YouTube channels from the topology of their observable public subscription networks.

The repository implements the pipeline proposed in the accompanying research paper:

1. collect public subscription relationships through YouTube Data API v3;
2. construct one directed network per reference channel;
3. compute a nine-dimensional structural signature;
4. transform and normalize the features;
5. calculate pairwise cosine similarities;
6. visualize networks, structural features, and the similarity matrix; and
7. export data, metrics, GEXF graphs, HTML visualizations, and a report.

## Structural signature

Each network is represented by:

- number of nodes;
- number of edges;
- density;
- average clustering coefficient;
- modularity;
- mean degree centrality;
- mean closeness centrality;
- mean betweenness centrality; and
- PageRank centralization.

PageRank centralization is used instead of mean PageRank because PageRank values sum to one, making their mean redundant with the number of nodes.

## Installation

Python 3.11 or newer is recommended.

```bash
git clone https://github.com/YOUR-USER/youtube-topology-comparator.git
cd youtube-topology-comparator

python -m venv .venv
source .venv/bin/activate              # Linux/macOS
# .venv\Scripts\activate               # Windows

pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` and select **Demonstration** to inspect the complete workflow without an API key.

## YouTube API

1. Create a project in Google Cloud Console.
2. Enable **YouTube Data API v3**.
3. Create an API key.
4. Select **YouTube API** in the application and enter the key in the password field.
5. Enter at least two channel handles, channel URLs, or channel IDs, one per line.

The API exposes only public subscription lists. An empty or sparse network may reflect channel privacy settings rather than an absence of subscriptions.

The collector uses the documented `subscriptions.list` `channelId` filter. Public-data requests use an API key; inaccessible lists are recorded as collection warnings instead of being interpreted as empty social relationships.

## CSV mode

Upload one CSV file per reference channel. Two columns are required:

```text
source_channel_id,target_channel_id
```

Optional columns are:

```text
source_channel_title,target_channel_title,target_description,
target_published_at,depth,collected_at
```

The filename (without `.csv`) becomes the network name.

## Normalization modes

- `auto`: z-score for three or more networks; max-absolute scaling for two;
- `zscore`: the paper's collection-level standardization;
- `maxabs`: recommended for a direct two-channel comparison;
- `none`: transformed features without feature-wise scaling.

Node and edge counts are transformed with `log(1+x)` before normalization. Constant features are removed from the similarity calculation and retained in the raw export.

Z-score normalization is not recommended for exactly two networks because every varying feature becomes a symmetric pair. The application reports this limitation explicitly.

## Tests

```bash
pytest -q
```

## Docker

```bash
docker build -t youtube-topology-comparator .
docker run --rm -p 8501:8501 youtube-topology-comparator
```

## Exported analysis

The downloadable ZIP contains:

- collected edge lists;
- raw node metrics;
- raw, transformed, and normalized structural signatures;
- cosine-similarity matrix;
- GEXF networks;
- standalone interactive HTML visualizations;
- execution metadata and collection warnings; and
- an automatically generated Markdown report.

## Methodological scope

The system compares selected topological properties. It does not infer musical genre, political orientation, audience profile, or content similarity. These external characteristics belong to subsequent case-study interpretation.

## License

MIT
