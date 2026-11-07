# ADR-0002: LiRA on one leave-one-out model family

- **Status:** Accepted
- **Date:** 2026-11-04

## Context
LiRA (Carlini et al., S&P 2022) needs many reference models per example, some trained with it (IN)
and some without (OUT). Training separate target and shadow populations doubles the compute, and on
an Apple M1 the budget is a few hours per dataset. The paper's own evaluation trains one family of
models on random halves of a pool and lets each model play the target in turn.

## Decision
Train one family of `N` models per dataset (64 for CIFAR-10 and Purchase100), each on a random half
of the pool with balanced masks (every example is IN for exactly half of the models). For a target
model, the other `N - 1` models are its reference models. Scores use the logit-scaled confidence
`phi = log(p / (1 - p))`, per-example medians as the Gaussian centres, and an optional pooled
variance (`-fixed` variants). Results are averaged over 16 targets. The authors' released scoring is
transcribed in `attacks/lira_reference.py` and evaluated on the same signals as a parity check.

## Alternatives considered
- **Disjoint target and shadow populations** - closer to a deployed attacker, but half the
  reference models per unit of compute.
- **Raw loss or confidence instead of `phi`** - far from Gaussian, so the per-example Gaussian fit
  is poor; the paper reports large drops in TPR at low FPR.
- **Use the reference implementation directly** - it truncates IN and OUT sets to the smallest count
  and uses a global variance; we keep it as a reference row in every table instead.

## Consequences
- 63 reference models per target, and the scaling curve (1 to 63 references) comes for free.
- The attacker knows the training recipe and the pool exactly (the paper's strong threat model);
  numbers are an upper bound for a less informed attacker.
- One training run per dataset feeds every attack, so the comparison between attacks is paired.
