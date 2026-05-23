#!/usr/bin/env python3
"""
Track A: Dynamic Uncertainty-Aware Attribution
Demo script — token-level hallucination detection using:
  - Information Gain (IG)
  - KL Divergence (KL)
  - Confidence Drop (CD)
  - Semantic Entropy (SE)

Usage:
    python demo.py --context "retrieved context" --response "generated response"

Example:
    python demo.py \
        --context "The Eiffel Tower was built between 1887 and 1889." \
        --response "The Eiffel Tower was built in 1900."
"""

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
MAX_LEN    = 2048
MIN_RESP   = 50

# ── Load models ────────────────────────────────────────────────
print("Loading Mistral-7B-v0.1...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
model.config.use_cache = False
model.eval()

print("Loading sentence encoder for Semantic Entropy...")
sem_model = SentenceTransformer('all-MiniLM-L6-v2')
print("All models loaded.\n")

# ── Core functions ─────────────────────────────────────────────
def get_logits(ctx, resp, max_len=MAX_LEN, min_resp_tokens=MIN_RESP):
    """
    Run Mistral-7B and return logits for response tokens only.
    Called twice: once with context, once without.
    Left-truncates context to always protect response tokens.
    """
    prefix_text      = f"Context: {ctx}\nResponse: "
    resp_only_ids    = tokenizer(resp, add_special_tokens=False)['input_ids']
    resp_token_count = min(len(resp_only_ids), max_len - min_resp_tokens - 10)
    prefix_ids       = tokenizer(prefix_text, add_special_tokens=True)['input_ids']
    budget           = max_len - resp_token_count - 1
    if len(prefix_ids) > budget:
        prefix_ids = prefix_ids[:1] + prefix_ids[-(budget - 1):]
    full_ids   = (prefix_ids + resp_only_ids[:resp_token_count])[:max_len]
    prefix_len = len(prefix_ids)
    input_ids  = torch.tensor([full_ids]).to(model.device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    logits = out.logits[0].detach().float().cpu()
    del out
    torch.cuda.empty_cache()
    logits     = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)[:-1]
    target_ids = input_ids[0][1:].detach().cpu()
    resp_start = prefix_len - 1
    return logits[resp_start:], target_ids[resp_start:]


def _compute_entropy(logits):
    """
    Internal helper for compute_ig only.
    Not displayed as a metric — entropy is Baseline 1, not a Track A metric.
    """
    probs = torch.softmax(logits, dim=-1)
    return (-(probs * torch.log(probs + 1e-10)).sum(dim=-1)).cpu().numpy()


def compute_ig(lw, lo):
    """
    Information Gain = H(without_context) - H(with_context) per token.
    LOW IG = context not helping = hallucination risk.
    Aggregated using mean_bot20 (bottom 20% most suspicious tokens).
    """
    ent_with    = _compute_entropy(lw)
    ent_without = _compute_entropy(lo)
    n = min(len(ent_with), len(ent_without))
    return ent_without[:n] - ent_with[:n]


def compute_kl(lw, lo, k=100):
    """
    KL(P_with || P_without) per token using top-100 tokens for stability.
    LOW KL = distributions similar = context ignored = hallucination risk.
    Aggregated using mean_bot20.
    """
    n     = min(lw.shape[0], lo.shape[0])
    p     = torch.softmax(lw[:n], dim=-1)
    q     = torch.softmax(lo[:n], dim=-1)
    idx   = torch.topk(p, k, dim=-1).indices
    p_top = torch.gather(p, -1, idx)
    q_top = torch.gather(q, -1, idx)
    return (p_top * (
        torch.log(p_top + 1e-10) - torch.log(q_top + 1e-10)
    )).sum(dim=-1).cpu().numpy()


def compute_conf_drop(logits):
    """
    Confidence Drop = max_prob(t) - max_prob(t+1) per token.
    HIGH CD = unstable predictions = hallucination risk.
    Aggregated using mean_top20.
    """
    top_probs = torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().numpy()
    if len(top_probs) < 2:
        return np.array([0.0])
    return top_probs[:-1] - top_probs[1:]


def compute_semantic_entropy(ctx, resp, n=3):
    """
    Semantic Entropy (Kuhn et al. 2023):
    Generate n responses at temperature=1.2, cluster by embedding similarity
    (cosine > 0.85 = same semantic cluster), entropy over cluster distribution.
    HIGH SE = diverse meanings = uncertain = hallucination risk.
    """
    prompt      = f"Context: {ctx}\nResponse:"
    generations = []

    for _ in range(n):
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=512
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=1.2,
                top_p=0.95,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id
            )
        text     = tokenizer.decode(gen[0], skip_special_tokens=True)
        gen_resp = text.split("Response:")[-1].strip()
        if len(gen_resp) > 5:
            generations.append(gen_resp)
        del gen, inputs
        torch.cuda.empty_cache()

    if len(generations) < 2:
        return 0.0

    embeddings = sem_model.encode(generations, convert_to_numpy=True)
    clusters   = []
    for i, emb in enumerate(embeddings):
        assigned = False
        for j in range(i):
            sim = np.dot(emb, embeddings[j]) / (
                np.linalg.norm(emb) * np.linalg.norm(embeddings[j]) + 1e-10
            )
            if sim > 0.85:
                clusters.append(clusters[j])
                assigned = True
                break
        if not assigned:
            clusters.append(i)

    _, counts = np.unique(clusters, return_counts=True)
    probs     = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def aggregate(arr, mode='mean_top20'):
    """
    Aggregate token-level scores to sample-level score.
    mean_bot20: bottom 20% — IG, KL (low = risk)
    mean_top20: top 20%    — CD (high = risk)
    """
    arr = np.array(arr, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    s   = np.sort(arr)
    n20 = max(1, int(0.2 * len(s)))
    if mode == 'mean_top20':
        return float(s[-n20:].mean())
    elif mode == 'mean_bot20':
        return float(s[:n20].mean())
    return float(s.mean())


# ── Main demo ──────────────────────────────────────────────────
def run_demo(context, response):
    print('=' * 65)
    print('  HALLUCINATION DETECTION — Track A')
    print('  Dynamic Uncertainty-Aware Attribution')
    print('  Model: Mistral-7B-v0.1')
    print('  Metrics: IG | KL Divergence | Confidence Drop | Semantic Entropy')
    print('=' * 65)
    print(f'\nContext:  {context[:200]}')
    print(f'\nResponse: {response[:200]}')
    print()

    # ── Step 1: Two forward passes ────────────────────────────
    print("Running two forward passes...")
    lw, _ = get_logits(context, response)  # with context
    lo, _ = get_logits("",      response)  # without context
    n      = min(lw.shape[0], lo.shape[0])
    lw, lo = lw[:n], lo[:n]
    print(f"Response tokens processed: {n}\n")

    # ── Step 2: Token-level metrics ───────────────────────────
    ig_arr = compute_ig(lw, lo)
    kl_arr = compute_kl(lw, lo)
    cd_arr = compute_conf_drop(lw)

    # ── Step 3: Semantic Entropy ──────────────────────────────
    print("Computing Semantic Entropy (generating n=3 samples)...")
    se_score = compute_semantic_entropy(context, response, n=3)
    print(f"Semantic Entropy: {se_score:.4f}\n")

    # ── Step 4: Response tokens for display ───────────────────
    resp_only_ids    = tokenizer(response, add_special_tokens=False)['input_ids']
    resp_token_count = min(len(resp_only_ids), MAX_LEN - MIN_RESP - 10)
    tokens           = [tokenizer.decode([t]) for t in resp_only_ids[:resp_token_count]]
    n_tok = min(len(tokens), len(ig_arr), len(kl_arr), len(cd_arr))

    # ── Step 5: Token-level table ─────────────────────────────
    print(f"{'Token':<20} {'KL':>9} {'IG':>9} {'CD':>9}  {'Flag'}")
    print('-' * 55)

    ig_threshold = -0.08
    flagged = 0
    for i in range(n_tok):
        tok  = repr(tokens[i])[:18]
        flag = '⚠' if ig_arr[i] < ig_threshold else ''
        if flag:
            flagged += 1
        print(f"{tok:<20} {kl_arr[i]:>9.4f} "
              f"{ig_arr[i]:>9.4f} {cd_arr[i]:>9.4f}  {flag}")

    # ── Step 6: Sample-level scores ───────────────────────────
    ig_score = aggregate(ig_arr, 'mean_bot20')
    kl_score = aggregate(kl_arr, 'mean_bot20')
    cd_score = aggregate(cd_arr, 'mean_top20')

    print()
    print(f"{'Metric':<22} {'Score':>10}  {'Direction':<12}  {'Interpretation'}")
    print('-' * 72)
    print(f"{'Information Gain':<22} {ig_score:>10.4f}  {'↓ low=risk':<12}  "
          f"low = context not helping")
    print(f"{'KL Divergence':<22} {kl_score:>10.4f}  {'↓ low=risk':<12}  "
          f"low = context ignored")
    print(f"{'Confidence Drop':<22} {cd_score:>10.4f}  {'↑ high=risk':<12}  "
          f"high = unstable predictions")
    print(f"{'Semantic Entropy':<22} {se_score:>10.4f}  {'↑ high=risk':<12}  "
          f"high = diverse meanings = uncertain")

    # ── Step 7: Verdict ───────────────────────────────────────
    print()
    print('=' * 65)
    if ig_score < -0.08:
        risk, symbol, reason = "HIGH",   "⚠ ", "context not helping — likely hallucinated"
    elif ig_score < -0.03:
        risk, symbol, reason = "MEDIUM", "? ", "weak context signal — borderline"
    else:
        risk, symbol, reason = "LOW",    "✓ ", "context used — likely faithful"

    print(f"  Hallucination Risk: {symbol} {risk}")
    print(f"  Reason: {reason}")
    print()
    print(f"  Key signals:")
    print(f"    Information Gain: {ig_score:>8.4f}  (threshold < -0.08 = HIGH)")
    print(f"    Flagged tokens:   {flagged:>8}  (tokens where IG < -0.08)")
    print(f"    KL Divergence:    {kl_score:>8.4f}  (lower = context less used)")
    print(f"    Confidence Drop:  {cd_score:>8.4f}  (higher = more unstable)")
    print(f"    Semantic Entropy: {se_score:>8.4f}  (higher = more uncertain)")
    print('=' * 65)

    torch.cuda.empty_cache()

    return {
        'ig_score':       ig_score,
        'kl_score':       kl_score,
        'cd_score':       cd_score,
        'se_score':       se_score,
        'risk_level':     risk,
        'flagged_tokens': flagged,
        'token_scores': {
            'tokens': tokens[:n_tok],
            'ig':     ig_arr[:n_tok].tolist(),
            'kl':     kl_arr[:n_tok].tolist(),
            'cd':     cd_arr[:n_tok].tolist(),
        }
    }


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Track A: Dynamic Uncertainty-Aware Attribution\n"
            "Token-level hallucination detection using IG, KL, CD, SE."
        )
    )
    parser.add_argument(
        '--context',  type=str, required=True,
        help='Retrieved context passage'
    )
    parser.add_argument(
        '--response', type=str, required=True,
        help='Generated response to evaluate'
    )
    args = parser.parse_args()
    run_demo(args.context, args.response)
