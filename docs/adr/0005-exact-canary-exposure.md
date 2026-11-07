# ADR-0005: Exact canary exposure by enumeration on a small language model

- **Status:** Accepted
- **Date:** 2026-11-06

## Context
Exposure (Carlini et al., USENIX Security 2019) is `log2 |R| - log2 rank` of an inserted secret among
all candidates. The paper estimates the rank from a sample of candidates with a skew-normal fit,
because enumerating the space is expensive for large models. Training-data extraction needs a model
small enough to fine-tune several times (with and without DP-SGD) within the 8 GB memory budget.

## Decision
Fine-tune Pythia-70M (deduplicated) on half of the WikiText-2 training paragraphs with 24 fictional
canaries ("the vault code of <fictional name> is d d d d d d") inserted 1 to 32 times. Compute every
canary's rank exactly by scoring all 10^6 six-digit continuations (a digit trie: the model runs on
all `10^d` partial secrets at each depth `d < 6` and next-digit log-probabilities are accumulated in
float64, 111,111 forward passes in batches of 4,096), and report the skew-normal estimate from 5,000 sampled
candidates next to it. Train only the linear layers in the transformer blocks so the hook-based
DP-SGD engine applies unchanged.

## Alternatives considered
- **Skew-normal estimate only** - cheap, but its accuracy at high exposure is exactly what is in
  question; computing both measures the gap.
- **A larger model (Qwen2.5-0.5B, SmolLM2-360M)** - more memorisation, but several fine-tuning runs
  and exhaustive enumeration would not fit the time and memory budget.
- **Real secrets or real personal data** - never; every canary is fictional.

## Consequences
- Exposure numbers are exact for this model, and the estimator's bias is measured, not assumed.
- Freezing embeddings and norms changes the fine-tuning recipe slightly; it is the same for the
  non-private and DP runs, so the comparison is fair.
