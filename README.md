# Track A: Dynamic Uncertainty-Aware Attribution
## Hallucination Detection via Token-Level Information Gain in RAG

**Course:** CS F429 Natural Language Processing — BITS Pilani Dubai — Semester 2 2025-26

**Team:**
- Abrar Akbar Mulla — 2023A7PS0173U
- Sherwin Gummadi — 2023A7PS0287U
- Prisha Bansal — 2023A7PS0273U
- Sarvagya Jain — 2023A7PS0318U

---

## Method Overview

We detect hallucination in RAG systems by running Mistral-7B-v0.1 twice per sample:
- **Pass 1:** Context + Response → logits_with
- **Pass 2:** Response only → logits_without

The difference between these distributions measures how much the model uses the retrieved context. Four metrics quantify this signal: Information Gain, KL Divergence, Confidence Drop, and Semantic Entropy. Weights and directions are derived from 500 train samples of RAGTruth and frozen for test.

**Key results:**
- RAGTruth test AUROC: **0.6712** (1000 samples, both baselines beaten)
- HaluEval zero-shot AUROC: **0.8579** (500 samples, no refitting)
- SOTA gap closed: **36.6%** toward supervised SOTA (fully unsupervised)

---

## Setup

### Requirements
- Python 3.10+
- CUDA GPU (16GB+ recommended — tested on L4 and A100)
- HuggingFace account with Mistral-7B access

### Installation
```bash
git clone https://github.com/[your-repo-url]
cd [repo-name]
pip install -r requirements.txt
```

### HuggingFace Login
```bash
huggingface-cli login
# Enter your HF token when prompted
# Token needs read access to mistralai/Mistral-7B-v0.1
```

---

## Running the Demo

```bash
python demo.py \
    --context "The Eiffel Tower was built between 1887 and 1889 by Gustave Eiffel." \
    --response "The Eiffel Tower was built in 1900 by Gustave Eiffel."
```

### Output
The script produces:
1. Token-level table showing KL Divergence, Information Gain, and Confidence Drop per token
2. Semantic Entropy at sample level
3. Aggregated scores for all 4 metrics
4. Hallucination verdict: HIGH ⚠ / MEDIUM ? / LOW ✓ with explanation

---

## Repository Structure
├── demo.py                              ← run this for hallucination detection
├── README.md                            
├── requirements.txt                     ← dependencies
├── Track_A_NLP_Assignment_Final.ipynb   ← full experiment notebook
└── results/
    ├── results_train_500.json           ← train pipeline scores
    ├── results_test_1000.json           ← test pipeline scores
    ├── halu_results.json                ← HaluEval scores
    ├── train_calibration.json           ← frozen directions + weights
    ├── df_test_scored.json              ← scored test dataframe
    └── exp3_temporal_precedence.png     ← Exp 3 line plot
---

## Reproducing Results

Open `Track_A_NLP_Assignment_Final.ipynb` in Google Colab with a GPU runtime and run cells in order.

Pre-computed results are in `results/` — load them to skip the pipeline (~16 hours of compute).

**Full pipeline runtimes:**
- Train pipeline (500 samples): ~4 hours on L4
- Test pipeline (1000 samples): ~12 hours on L4
- HaluEval pipeline (500 samples): ~3 hours on A100

---

## Metrics

| Metric | Formula | Signal | Aggregation |
|--------|---------|--------|-------------|
| Information Gain | H(without) - H(with) | Low = context ignored | mean_bot20 |
| KL Divergence | KL(P_with ∥ P_without) | Low = context ignored | mean_bot20 |
| Confidence Drop | max_prob(t) - max_prob(t+1) | High = unstable | mean_top20 |
| Semantic Entropy | Entropy over semantic clusters | High = uncertain | sample-level |

---

## Academic Integrity

This project uses generative AI tools for code assistance and grammar correction only, as per BITS Pilani Gen AI Usage Guidelines (effective 1 April 2026).
