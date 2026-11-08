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

## Standards mapping

The attacks and the project's own threats are mapped to the OWASP Top 10 for LLM Applications
2026 and MITRE ATLAS v2026.10 (IDs and names checked against `ATLAS.yaml` from the
[v2026.10 release of mitre-atlas/atlas-data](https://github.com/mitre-atlas/atlas-data/releases)).
The OWASP Top 10 for Agentic Applications 2026 is not mapped because no agent is involved.

| Threat | OWASP LLM 2026 | MITRE ATLAS v2026.10 | Role of this project |
|---|---|---|---|
| Membership inference (loss, calibrated loss, shadow models, LiRA, sequence MIA) | LLM02:2026 Sensitive Information Disclosure | AML.T0137.000 Training Data Information from AI Models: Infer Membership; AML.T0040 AI Model Inference API Access | Implements the attacks and measures TPR at low FPR |
| Shadow and reference models trained from the same recipe | LLM02:2026 Sensitive Information Disclosure | AML.T0005.000 Replicate AI Model Behavior: Replicate Behavior from Gathered AI Artifacts | Implements, as part of the attacks above |
| Canary exposure, greedy and untargeted extraction | LLM02:2026 Sensitive Information Disclosure | AML.T0137.003 Training Data Information from AI Models: Extract Memorized Data; AML.T0024 Exfiltration via AI Inference API | Implements and measures on Pythia-70M |
| DP-SGD as the defence | LLM02:2026 Sensitive Information Disclosure | AML.M0041 Privacy-Preserving AI Training and Inference | Implements, with accountants checked against Opacus |
| Malicious dataset archive | LLM04:2026 Supply Chain | AML.T0010.002 Supply Chain Compromise: Data Repository; AML.T0011.000 User Execution: Unsafe AI Artifacts | Mitigates with a restricted unpickler and pinned hashes (AML.M0014 Verify AI Artifacts) |
| Untrusted model weights | LLM04:2026 Supply Chain | AML.T0011.000 User Execution: Unsafe AI Artifacts | Mitigates: only files the benchmarks write, and safetensors (AML.M0011 Restrict Library Loading) |
| Oversized requests | LLM06:2026 Unbounded Consumption | AML.T0029 Denial of AI Service | Mitigates with body and list limits (AML.M0036 Limit AI Workload Resource Consumption) |
| Malicious dependency | LLM04:2026 Supply Chain | AML.T0010.001 Supply Chain Compromise: Software | Mitigates with a lockfile and CodeQL (AML.M0016 Vulnerability Scanning) |

In NIST AI 600-1 (Generative AI Profile) terms the project measures the Data Privacy risk.

## Residual risk
- Empirical attacks give lower bounds on leakage; a model that resists these attacks is not private
  unless it was trained with DP-SGD and correct accounting.
- The service binds to 127.0.0.1 by default and has no authentication; expose it only behind one.
