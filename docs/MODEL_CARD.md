# Model Card - Privacy Attacks Lab

## Overview
- **What it is:** a toolkit and a set of measured experiments: membership inference attacks, DP-SGD
  with RDP, PRV and GDP accounting, canary exposure and extraction. The models are research
  artefacts trained only to be attacked.
- **Models trained:**
  - CIFAR-10: a GroupNorm CNN (0.11M parameters), 64 models, 15 epochs on 25,000 images each.
  - Purchase100: an MLP 600-512-256-128-100 (0.48M parameters), 64 models on 98,662 records each,
    and DP-SGD families at epsilon 1 and 8.
  - Language model: Pythia-70M (deduplicated, EleutherAI) fine-tuned on WikiText-2 with 24
    fictional canaries, without DP and with DP-SGD at epsilon 8 and 1.
- **Version and date:** v0.1.0, November 2026.

## Intended use
- Measuring privacy leakage of small models and the protection DP-SGD gives, and teaching the
  methods with readable implementations checked against references.
- Users: ML engineers, privacy and security researchers, reviewers of DP claims.

## Out of scope
- Attacking models or services you do not own or have permission to test.
- Treating these models as useful classifiers or language models; they are deliberately small.
- Certifying a production system as private from empirical attack results alone.

## Performance
See [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md) for model accuracy, attack TPR at 0.1% and 1% FPR,
privacy/utility curves and canary exposure, all with seeds or bootstrap intervals.

## Risks and limitations
- Dual use: membership inference and extraction can be pointed at other models. The code only
  attacks models it trains itself on public data, and the canaries are fictional.
- Results depend on the training recipe; better-regularised or larger models leak differently.

## Safety mitigations
- No real secrets or personal data: canaries are synthetic and the datasets are public benchmarks.
- DP-SGD is provided as the defence, with accounting validated against Opacus.

## Compute footprint
Everything ran on an Apple M1 (8 cores, 8 GB, MPS) shared with other work; per-run timings are in
the result files under `benchmarks/results/`.
