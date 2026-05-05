# TextGuard 2.0

Hierarchical attention for compliance risk detection in development-bank
documents. This repository contains the code, splits, and evaluation
scripts for the paper *"TextGuard 2.0: Hierarchical Attention for
Compliance Risk Detection in Development-Bank Documents"*.

## What this is

Compliance review of Inter-American Development Bank (IDB) project
documents is slow and inconsistent. TextGuard 2.0 is a one-level
hierarchical attention network (HAN-CE) that reads paragraph-scale
chunks of IDB project PDFs with DistilBERT, learns chunk-level attention
weights, and aggregates them into a document-level risk classifier. The
attention weights also serve as reviewer-readable evidence — pointing
to the specific passages that drove each prediction.

The repository covers four model configurations from the paper's
ablation:

- **Flat BERT (TG 1.0)** — chunk-level fine-tuning, leaky chunk-split eval
- **HAN-NoAttn** — hierarchy with mean-pooling, document-grouped eval
- **HAN-CE (ours)** — hierarchy with learned attention, cross-entropy
- **HAN-C** — HAN-CE plus focal loss + Spanish back-translation
  (negative result, included for the ablation)

It also covers a multi-label extension (Financial / Social /
Environmental risk) using the same backbone.

## Headline results

All numbers are from the paper, reported on the document-grouped test
split (22 held-out documents, fixed seed 42).

### Risk task (Table 1 in paper)

| Model | Acc | Macro-P | Macro-F1 | Risk recall | No-risk recall |
|---|---|---|---|---|---|
| Flat BERT (TG 1.0, leaky) | 0.910 | 0.751 | 0.710 | 0.91 | 0.50 |
| HAN-NoAttn (mean-pool) | 0.792 | 0.701 | 0.683 | 0.97 | 0.31 |
| **HAN-CE (ours)** | **0.9545** | **0.835** | **0.820** | **1.00** | **0.54** |
| HAN-C (focal + ES BT) | 0.748 | 0.694 | 0.659 | 0.93 | 0.31 |

### Multi-label risk categorisation (Table 3 in paper)

| Category | Precision | Recall | F1 |
|---|---|---|---|
| Financial | 0.79 | 0.79 | 0.79 |
| Social | 0.98 | 0.62 | 0.76 |
| Environmental | 0.75 | 0.55 | 0.63 |
| Macro avg | 0.85 | 0.65 | 0.73 |

**Key finding:** adding focal loss and back-translation to HAN-CE
*regresses* macro-F1 by 16 points (0.82 → 0.66). The paper diagnoses
three reasons (focal-loss hyperparameter brittleness in low-data text,
label drift from back-translating compliance vocabulary, and
architectural redundancy with attention-based minority emphasis).
HAN-C is included here so the ablation can be reproduced, not as a
recommended configuration.

## Repository contents

```
.
├── NLP_Pipeline_GoogleStyle.ipynb    # main notebook — full pipeline
├── requirements.txt                  # pinned dependencies
├── data/
│   └── Labeled_docs_data.csv         # chunk-level labels (see Data section)
├── splits/
│   └── doc_grouped_seed42.json       # 87 train / 22 test document IDs
├── paper/
│   └── neurips_2026.pdf              # camera-ready paper
└── README.md
```

## Setup

Tested on Python 3.10 with a single NVIDIA T4 GPU (Google Colab) and on
Ubuntu 22.04 with an A100. CPU-only execution works but is slow
(~10× slower than T4).

```bash
git clone git@github.com:Sai-Srinivas7/TextGuard-2.0.git
cd TextGuard-2.0
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

`requirements.txt` (pinned to versions we tested):

```
torch==2.1.0
transformers==4.36.2
datasets==2.16.1
evaluate==0.4.1
accelerate==0.26.1
scikit-learn==1.3.2
spacy==3.7.2
PyMuPDF==1.23.8
python-docx==1.1.0
sentencepiece==0.1.99
sacremoses==0.1.1
matplotlib==3.8.2
seaborn==0.13.1
pandas==2.1.4
numpy==1.26.3
```

## Data

The labelled corpus is 109 IDB project PDFs spanning 17 sectors,
producing 956 chunks at most 10 chunks per document (128 tokens each).
Class imbalance is 6.4:1 toward the *risk* class.

The IDB project documents themselves are publicly disclosed by the
Bank but we do not redistribute them in this repository while we
confirm derivative-distribution rights. To reproduce:

1. Place `Labeled_docs_data.csv` (chunk-level labels) at
   `data/Labeled_docs_data.csv`. The CSV schema is:
   ```
   document_id, chunk_id, chunk_text, risk_label, compliance_label,
   financial_risk, social_risk, environmental_risk
   ```
2. The fixed train/test split (87/22 documents, seed 42) is in
   `splits/doc_grouped_seed42.json`. The notebook loads this directly
   so the split is deterministic across machines.

If you don't have the CSV, contact the authors.

## How to reproduce the paper

The whole pipeline runs from one notebook. Pick one:

**Option A — interactive:**
```bash
jupyter notebook NLP_Pipeline_GoogleStyle.ipynb
```
Run all cells top to bottom.

**Option B — headless:**
```bash
jupyter nbconvert --to notebook --execute \
  NLP_Pipeline_GoogleStyle.ipynb \
  --output NLP_Pipeline_executed.ipynb \
  --ExecutePreprocessor.timeout=-1
```

Total wall time on a single T4: roughly 2–3 hours for the full
ablation (Flat BERT + HAN-NoAttn + HAN-CE + HAN-C + multi-label).
Total compute reported in the paper is under 12 GPU-hours.

### Hyperparameters (from paper Section 5.3)

| Setting | Value |
|---|---|
| Encoder | `distilbert-base-uncased` (66M params, all fine-tuned) |
| Max chunk length | 128 tokens |
| Max chunks per document | 10 |
| Optimiser | AdamW |
| Learning rate | 2 × 10⁻⁵ |
| Weight decay | 0.01 |
| Batch size | 4 |
| Epochs | 3 |
| Splitter | `GroupShuffleSplit`, seed 42 |
| Focal loss (HAN-C only) | α = 0.25, γ = 2.0 |
| Back-translation (HAN-C only) | MarianMT EN → ES → EN, minority chunks |

### What you should see

Numbers should match the paper's Table 1 within ±0.01 on each metric
on a single fixed-seed run. Larger drift usually means a different
`transformers` version reshaping tokeniser output — pin to the
versions in `requirements.txt`.

## Method (in one paragraph)

Each document is parsed into up to 10 paragraph-scale chunks. Each
chunk is independently encoded by DistilBERT and represented by its
`[CLS]` embedding `e_i ∈ R^768`. A two-layer MLP attention module
produces softmax weights `α_i` over the chunks, and the document
vector `d = Σ α_i e_i` feeds a linear classifier. All DistilBERT
parameters are fine-tuned end-to-end. The attention weights are kept
at inference time and surfaced as evidence — the top-attention chunk
points the reviewer to the passage most responsible for the
prediction. See Figure 1 in the paper.

## Citation

```bibtex
@inproceedings{textguard2026,
  title={TextGuard 2.0: Hierarchical Attention for Compliance Risk
         Detection in Development-Bank Documents},
  author={Ramamurthy, Chandan Kumar and Guttikonda, Sai Srinivas and
          Raghuthaman, Anaswara},
  year={2026}
}
```

## Licenses for existing assets used

- DistilBERT: Apache 2.0 (Sanh et al., 2019)
- MarianMT EN↔ES: Apache 2.0, Helsinki-NLP `opus-mt` family
- HuggingFace Transformers: Apache 2.0
- PyTorch: BSD 3-Clause
- scikit-learn: BSD 3-Clause
- spaCy: MIT
- IDB project documents: publicly disclosed by the Inter-American
  Development Bank

## Limitations

The paper is honest about these and so is this README:

- 109 documents is modest. A single anomalous test document can move
  macro-F1 by 2–3 points.
- Results are from a single document-grouped split, not k-fold CV.
- The 6.4:1 imbalance is moderate; the focal-loss negative result
  should not be extrapolated to extreme (50:1+) imbalance regimes.
- Spanish was the only back-translation pivot we tested.

## Contact

Open an issue on this repo, or email the authors at the addresses on
the paper.
