# Track A: Dynamic Uncertainty-Aware Attribution
## Hallucination Detection via Token-Level Information Gain in RAG

**Course:** CS F429 Natural Language Processing — BITS Pilani Dubai — Semester 2 2025-26  
**Team Number:** 2  
**Track:** A — Dynamic Uncertainty-Aware Attribution  

**Team Members:**
| Name | ID |
|------|----|
| Abrar Akbar Mulla | 2023A7PS0173U |
| Sherwin Gummadi | 2023A7PS0287U |
| Prisha Bansal | 2023A7PS0273U |
| Sarvagya Jain | 2023A7PS0318U |

---

## Method Overview

We detect hallucination in RAG systems by running Mistral-7B-v0.1 twice per sample:

- **Pass 1:** `Context + Response → Mistral-7B → logits_with`
- **Pass 2:** `Response only → Mistral-7B → logits_without`

The difference between these distributions measures how much the model uses the retrieved context. When a model hallucinates, it ignores the context — so the two distributions are similar. When it is faithful, the context shifts the distributions significantly.

Four metrics quantify this signal:

| Metric | Formula | Signal | Aggregation |
|--------|---------|--------|-------------|
| Information Gain | IG = H(without) − H(with) | Low = context ignored | mean_bot20 |
| KL Divergence | KL(P_with ∥ P_without) | Low = context ignored | mean_bot20 |
| Confidence Drop | max_prob(t) − max_prob(t+1) | High = unstable | mean_top20 |
| Semantic Entropy | Entropy over semantic clusters of n=3 generations | High = uncertain | sample-level |

Weights and directions are derived from 500 RAGTruth train samples and frozen for test — the test set is never used to set them.

**Key Results:**
| Dataset | AUROC | Samples |
|---------|-------|---------|
| RAGTruth test set | **0.6712** | 1000 |
| HaluEval zero-shot | **0.8579** | 500 |
| SOTA gap closed | **36.6%** | — |

---

## Repository Structure
NLP-TrackA-RAG-Hallucination-Detection/
│
├── demo.py                              # Main demo script — run this
├── README.md                            # This file
├── requirements.txt                     # All dependencies
├── Track_A_NLP_Assignment_Final.ipynb   # Full experiment notebook
│
└── results/
├── results_train_500.json           # Train pipeline metric scores
├── results_test_1000.json           # Test pipeline metric scores
├── halu_results.json                # HaluEval zero-shot scores
├── train_calibration.json           # Frozen directions + weights
├── df_test_scored.json              # Scored test dataframe
└── exp3_temporal_precedence.png     # Experiment 3 line plot
---

## Setup

### System Requirements
- **Python version:** 3.10 or higher
- **GPU:** CUDA-enabled GPU with 16GB+ VRAM (tested on NVIDIA L4 and A100)
- **RAM:** 32GB+ recommended
- **Storage:** ~15GB for Mistral-7B model weights

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Dev-AbrarMulla/NLP-TrackA-RAG-Hallucination-Detection
cd NLP-TrackA-RAG-Hallucination-Detection
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

All required packages:
torch>=2.0.0
transformers>=4.35.0
sentence-transformers>=2.2.0
datasets>=2.14.0
scikit-learn>=1.3.0
scipy>=1.11.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
huggingface-hub>=0.19.0
accelerate>=0.24.0
bitsandbytes>=0.41.0

### Step 3 — HuggingFace Login
Mistral-7B-v0.1 requires a HuggingFace account with model access.

```bash
huggingface-cli login
```

Enter your HuggingFace token when prompted. Your token needs read access to `mistralai/Mistral-7B-v0.1`. You can generate a token at (https://huggingface.co/settings/tokens).

---

## Running the Demo

### Exact command to run demo.py:
```bash
python demo.py \
    --context "Your retrieved context passage here." \
    --response "The generated response to evaluate here."
```

### Example — Hallucinated response:
```bash
python demo.py \
    --context "The Amazon rainforest covers approximately 5.5 million square kilometres and produces around 20 percent of the worlds oxygen." \
    --response "The Amazon rainforest spans over 7 million square kilometres and produces 30 percent of the worlds oxygen supply."
```

### Example — Faithful response:
```bash
python demo.py \
    --context "The Eiffel Tower was built between 1887 and 1889 as the entrance arch for the 1889 Worlds Fair. It stands 330 metres tall and was designed by engineer Gustave Eiffel." \
    --response "The Eiffel Tower was designed by Gustave Eiffel and built between 1887 and 1889 for the Worlds Fair. It stands 330 metres tall."
```

### Output
The script produces:
1. **Token-level table** — KL Divergence, Information Gain, and Confidence Drop per response token, with ⚠ flags on suspicious tokens (IG < -0.08)
2. **Semantic Entropy** — sample-level score from n=3 generations at temperature=1.2
3. **Aggregated metric scores** — mean_bot20 for IG/KL, mean_top20 for CD
4. **Hallucination verdict** — HIGH ⚠ / MEDIUM ? / LOW ✓ with reasoning

### Expected runtime
- Token-level metrics (IG, KL, CD): ~10-30 seconds per sample
- Semantic Entropy (n=3 generations): ~2-3 additional minutes per sample
- Total per sample: ~3 minutes on A100

---

## Reproducing Full Experiments

Open `Track_A_NLP_Assignment_Final.ipynb` in Google Colab with a GPU runtime. Run cells in order. All experiments are documented with inline comments.

Pre-computed results are available in `results/` — load them to skip the pipeline and directly reproduce all tables and plots.

**Full pipeline compute times:**
| Pipeline | Samples | GPU | Time |
|----------|---------|-----|------|
| Train | 500 | L4 | ~4 hours |
| Test | 1000 | L4 | ~12 hours |
| HaluEval | 500 | A100 | ~3 hours |

---

## Academic Integrity

This project uses generative AI tools for code assistance and grammar correction only, as per BITS Pilani Gen AI Usage Guidelines (effective 1 April 2026). All analysis, interpretation, experimental design, and report writing is our own work.
