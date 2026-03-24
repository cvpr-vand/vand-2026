# VAND 4.0 Challenge @ CVPR 2026

Participant scaffold for the **Visual Anomaly and Novelty Detection (VAND) 4.0 Challenge** at CVPR 2026.

Two competition tracks:

| Track          | Dataset    | Task                             | Metric | Submission                                                   |
| -------------- | ---------- | -------------------------------- | ------ | ------------------------------------------------------------ |
| **Industrial** | MVTec AD 2 | Pixel-level anomaly segmentation | SegF1  | tar.gz → [benchmark.mvtec.com](https://benchmark.mvtec.com/) |
| **Retail**     | Kaputt 2   | Image-level defect scoring       | AP_any | CSV zip → [Codabench](https://www.codabench.org/)            |

## Repository Structure

```
vand-2026/
├── packages/vand/             # Shared framework library (dataloaders, evaluation, submission)
│   └── src/vand/
│       ├── industrial/        # MVTec AD 2 dataset, SegF1 evaluation, tar.gz packaging
│       └── retail/            # Kaputt/Kaputt2 dataset, AP evaluation, CSV packaging
│
├── tracks/                    # Participant stubs — implement your model here
│   ├── industrial/            # AnomalyModel skeleton + wired train/test/submit
│   └── retail/                # DefectModel skeleton + wired train/test/submit
│
├── examples/                  # Working baseline implementations for reference
│   ├── industrial/            # ResNet-18 + kNN feature-memory baseline
│   └── retail/                # ResNet-18 + cosine distance baseline
│
├── INDUSTRIAL_TRACK_RULES.md  # Full rules for Track 1
└── RETAIL_TRACK_RULES.md      # Full rules for Track 2
```

This is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/). Each subfolder under `packages/`, `tracks/`, and `examples/` is an independent Python package that shares the `vand` library via workspace dependencies.

## Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Quick Start

### 1. Install dependencies

```bash
uv sync --all-packages
```

### 2. Download data

- **Industrial**: Download MVTec AD 2 from [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad-2)
- **Retail**: Download Kaputt from [kaputt-dataset.com](https://www.kaputt-dataset.com/) and Kaputt 2 from [kaputt-dataset.com/kaputt2](https://www.kaputt-dataset.com/kaputt2)

### 3. Pick your track

**Industrial Track** — see [`tracks/industrial/README.md`](tracks/industrial/README.md)

```bash
# Implement your model in tracks/industrial/src/industrial/model.py, then:
uv run --project tracks/industrial train-industrial --data_root /path/to/mvtec_ad_2
uv run --project tracks/industrial test-industrial --data_root /path/to/mvtec_ad_2 --weights_dir ./weights
uv run --project tracks/industrial submit-industrial --predictions_dir ./predictions
```

**Retail Track** — see [`tracks/retail/README.md`](tracks/retail/README.md)

```bash
# Implement your model in tracks/retail/src/retail/model.py, then:
uv run --project tracks/retail train-retail --data_root /path/to/kaputt
uv run --project tracks/retail test-retail --data_root /path/to/kaputt2 --weights_path ./weights/model.pt
uv run --project tracks/retail submit-retail --predictions ./predictions.csv
```

### 4. Run a baseline (optional)

Working examples are provided under `examples/` for reference:

```bash
# Industrial baseline
uv run --project examples/industrial train-industrial-example --data_root /path/to/mvtec_ad_2

# Retail baseline
uv run --project examples/retail train-retail-example --data_root /path/to/kaputt
```

## Development

```bash
# Lint
uv run ruff check packages/ tracks/ examples/

# Format
uv run ruff format packages/ tracks/ examples/

# Type check
uv run mypy packages/vand/src/vand/
```

## Local Evaluation Disclaimer

The metric computation utilities in this repository (`vand.industrial.evaluate` and `vand.retail.evaluate`) are provided **for convenience and local testing only**. Official scores are computed by the respective challenge servers, whose implementations may differ. No claims or entitlements can be derived from local evaluation results.

## Rules

Read the full challenge rules before submitting:

- [Industrial Track Rules](INDUSTRIAL_TRACK_RULES.md)
- [Retail Track Rules](RETAIL_TRACK_RULES.md)

## License

Original code: Apache-2.0 (Intel Corporation)
Code derived from MVTec: CC-BY-NC-4.0 (MVTec Software GmbH)
