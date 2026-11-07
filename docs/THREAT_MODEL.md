# Threat Model - Privacy Attacks Lab

This project measures how much a trained model reveals about its training data, and how well DP-SGD
limits it. The threat model below is the one the attacks assume; the second part covers the
project's own service and code.

## Assets
- The training set of a model: which records were used (membership) and their content (extraction).
- Inserted secrets (canaries) standing in for sensitive strings in fine-tuning text.
- The integrity of the privacy accounting: a wrong epsilon silently invalidates every DP claim.

## Adversary (the attacks)
| Attack | Access | Knowledge | Goal |
|---|---|---|---|
| Loss threshold (Yeom et al. 2018) | target's loss on a record | label of the record | membership |
| Calibrated loss (Watson et al. 2022) | target's loss | reference models' average loss per record | membership |
| Shadow models (Shokri et al. 2017) | target's top-k confidences | training recipe, data from the same distribution | membership |
| LiRA online/offline (Carlini et al. 2022) | target's confidence on the true class | training recipe, the pool, many reference models (IN and OUT for online) | membership at low FPR |
| Sequence MIA (loss, Min-K%, zlib, reference) | token log-probabilities | base model (for the reference attack) | membership of a paragraph |
| Canary exposure and greedy extraction (Carlini et al. 2019) | token log-probabilities, generation | canary format and prefix | recover a secret |
| Untargeted extraction (Carlini et al. 2021) | sampling | base model for ranking | recover verbatim training text |

The LiRA setting is deliberately strong (the attacker knows the exact training recipe and the pool),
so the measured TPRs are upper bounds for a less informed attacker and lower bounds for a stronger one
(for example one that also sees intermediate checkpoints or gradients).

## Defence and its guarantees
DP-SGD with Poisson sampling, per-example clipping and calibrated Gaussian noise gives
(epsilon, delta)-DP for every record. Any membership test then satisfies
`TPR <= min(e^eps FPR + delta, 1 - e^-eps (1 - delta - FPR))` (Kairouz et al. 2015); the evaluation
plots measured attack TPRs against this bound. The guarantee holds only if the implementation
matches the analysis, hence: Poisson sampling (not shuffling), noise scaled by the expected batch
size, no BatchNorm, and accountants validated against Opacus.

## The service and code
| Threat | Vector | Impact | Mitigation |
|---|---|---|---|
| Malicious dataset archive | Tampered CIFAR-10 pickle | Code execution while loading | Restricted unpickler; SHA-256 pinned downloads |
| Oversized requests | Large JSON bodies to the HTTP service | Memory exhaustion | 8 MB body limit, step and list length limits, 400 on invalid input |
| Untrusted model weights | Loading third-party checkpoints | Code execution | Only `npz` and `torch.save` files written by the benchmarks; Hugging Face weights in safetensors |
| Supply chain | Malicious dependency | Code execution | Dependencies pinned in `uv.lock`; CodeQL on every push to main |

## Residual risk
- Empirical attacks give lower bounds on leakage; a model that resists these attacks is not private
  unless it was trained with DP-SGD and correct accounting.
- The service binds to 127.0.0.1 by default and has no authentication; expose it only behind one.
