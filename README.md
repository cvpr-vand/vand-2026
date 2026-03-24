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
├── tracks/                    # Participant stubs — implement your model here
│   ├── industrial/            # Model skeleton + train/test/submit entrypoints
│   │   └── src/industrial/
│   │       ├── model.py       # ← Implement your anomaly segmentation model
│   │       ├── train.py       # Training entrypoint
│   │       ├── test.py        # Inference entrypoint
│   │       └── submit.py      # Submission packaging
│   └── retail/                # Model skeleton + train/test/submit entrypoints
│       └── src/retail/
│           ├── model.py       # ← Implement your defect detection model
│           ├── train.py       # Training entrypoint
│           ├── test.py        # Inference entrypoint
│           └── submit.py      # Submission packaging
│
└── utils/                     # Shared helper utilities (dataloaders, evaluation, submission)
    ├── industrial/            # MVTec AD 2 dataset, SegF1 evaluation, tar.gz packaging
    ├── retail/                # Kaputt 2 dataset, AP evaluation, CSV packaging
    └── auto_batch.py          # Auto batch-size fitting decorator
```

## Prerequisites

- Python ≥ 3.11
- Dependencies listed in [`utils/requirements.txt`](utils/requirements.txt)

## Quick Start

### 1. Install dependencies

```bash
pip install -r utils/requirements.txt
```

### 2. Download data

- **Industrial**: Download MVTec AD 2 from [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad-2)
- **Retail**: Download Kaputt from [kaputt-dataset.com](https://www.kaputt-dataset.com/) and Kaputt 2 from [kaputt-dataset.com/kaputt2](https://www.kaputt-dataset.com/kaputt2)

### 3. Pick your track

**Industrial Track** — see [`tracks/industrial/README.md`](tracks/industrial/README.md)


**Retail Track** — see [`tracks/retail/README.md`](tracks/retail/README.md)


## Local Evaluation Disclaimer

The evaluation utilities in `utils/industrial/evaluate.py` and `utils/retail/evaluate.py` are provided **for convenience and local testing only**. Official scores are computed by the respective challenge servers, whose implementations may differ. No claims or entitlements can be derived from local evaluation results.

## License

Original code: Apache-2.0 (Intel Corporation)
Code derived from MVTec: CC-BY-NC-4.0 (MVTec Software GmbH)
