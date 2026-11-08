<h1 align="center">Privacy Attacks Lab</h1>

<p align="center">
  <em>Membership inference attacks (LiRA, shadow models, calibrated loss), DP-SGD with RDP and PRV accountants checked against Opacus, and canary exposure and extraction from small language models.</em>
</p>

<p align="center">
  <a href="https://github.com/adeoluo/privacy-attacks-lab/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/adeoluo/privacy-attacks-lab/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/adeoluo/privacy-attacks-lab/releases"><img alt="Release" src="https://img.shields.io/github/v/release/adeoluo/privacy-attacks-lab"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14%20%7C%203.15-blue">
  <img alt="Rust" src="https://img.shields.io/badge/rust-stable-orange">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

---

## Motivation

A model that has seen a record behaves slightly differently on it, and that difference can reveal
who was in the training set or reproduce text that was meant to stay private. The useful question
is not the average-case accuracy of such an attack but how many members it identifies with almost
no false positives, which is where real harm happens (Carlini et al. 2022). Answering it needs
many models trained on known subsets of the data, a careful attack, and ROC metrics read at 0.1%
and 1% false positive rate.

This lab implements the attacks (loss threshold, difficulty calibrated loss, shadow models, LiRA
online and offline), the defence (DP-SGD with per-example clipping and Poisson sampling) and the
privacy accounting behind its guarantee (RDP, PRV and GDP accountants, with RDP and ROC analysis in
a Rust core). Every accountant is checked against Opacus to many digits, the LiRA scores are checked
against the authors' released scoring code, and every number below comes from runs on the full
CIFAR-10 and Purchase100 datasets with 64 models each.

## Architecture

<p align="center">
  <img src="docs/architecture/architecture.svg" alt="Architecture diagram" width="860">
</p>

<sub>Legend: amber = entry points and external systems, indigo = core logic, green = data and state, blue = validation and control, grey = observability and outputs.</sub>

Benchmarks and the CLI choose a dataset preset; the shadow harness trains a family of models,
each on a seeded random half of the pool, records the membership masks and every model's logits
on every example (the run store), and the attacks score each target model using the others as
references (leave one out, [ADR-0002](docs/adr/0002-lira-leave-one-out-model-family.md)). DP-SGD
trains the same families with noise calibrated by the accountants. The Rust core computes ROC
curves, TPR at fixed FPR and bootstrap intervals; results land in `benchmarks/results/` as JSON
and the HTTP service exposes accounting, ROC metrics and exposure with Prometheus metrics.

## Results

All runs: Apple M1 (8 cores, 8 GB, MPS), shared with other work. Each dataset's pool is split at
random into halves 64 times; each model trains on its half, so every example is a member of about
32 models and a non-member of the other 32. Every model in turn is the target and the other 63 are
the attacker's reference models. Tables give the mean over 16 target models (standard deviation
in parentheses); per-target bootstrap intervals (1,000 resamples) are in the result files.

### Membership inference on CIFAR-10 (50,000 images, 64 CNNs)

GroupNorm CNN (0.11M parameters), 15 epochs on 25,000 images each, no augmentation in training;
train accuracy 98.5%, held-out accuracy 75.5%. Queries: the image and its mirror. 1.6 hours to
train all 64 models.

| Attack | AUC | TPR at 0.1% FPR | TPR at 1% FPR |
|---|---:|---:|---:|
| Loss threshold (Yeom et al. 2018) | 56.0% (0.3) | 0.10% (0.02) | 1.02% |
| Calibrated loss (Watson et al. 2022) | 64.5% (0.5) | 1.75% (0.27) | 6.10% |
| Shadow models (Shokri et al. 2017) | 71.9% (1.2) | 0.99% (0.17) | 5.68% |
| LiRA offline, fixed variance | 66.1% (0.7) | 0.91% (0.16) | 4.79% |
| LiRA online, per-example variance | 74.7% (0.9) | 3.27% (0.48) | 10.71% |
| **LiRA online, fixed variance** | **74.9% (0.8)** | **4.05% (0.76)** | **12.04%** |
| LiRA online, authors' scoring code | 74.8% (0.9) | 3.27% (0.51) | 10.78% |

### Membership inference on Purchase100 (197,324 records, 64 MLPs)

MLP 600-512-256-128-100 (0.48M parameters), 20 epochs on 98,662 records each; train accuracy
99.8%, held-out accuracy 93.0%. 34 minutes to train all 64 models.

| Attack | AUC | TPR at 0.1% FPR | TPR at 1% FPR |
|---|---:|---:|---:|
| Loss threshold | 54.6% (0.1) | 0.10% (0.01) | 1.00% |
| Calibrated loss | 62.4% (0.1) | 1.13% (0.10) | 4.69% |
| Shadow models | 58.1% (0.2) | 0.14% (0.02) | 1.34% |
| LiRA offline, fixed variance | 63.5% (0.1) | 1.02% (0.10) | 5.03% |
| LiRA online, per-example variance | 65.1% (0.1) | 1.30% (0.09) | 5.87% |
| **LiRA online, fixed variance** | **65.5% (0.1)** | **1.49% (0.14)** | **6.66%** |
| LiRA online, authors' scoring code | 65.2% (0.1) | 1.32% (0.09) | 5.90% |

Reproduce: `python benchmarks/train_shadows.py cifar10` then `python benchmarks/mia_benchmark.py
cifar10` (same for `purchase100`). Raw results: [`mia_cifar10.json`](benchmarks/results/mia_cifar10.json),
[`mia_purchase100.json`](benchmarks/results/mia_purchase100.json).

<p align="center">
  <img src="docs/figures/roc_log.png" alt="Log-scale ROC curves of every attack on CIFAR-10 and Purchase100" width="860">
</p>

### Defence: DP-SGD on Purchase100

The same pool and attacks against families trained with DP-SGD (Poisson sampling, expected batch
1,024, clipping norm 1, 30 epochs, noise calibrated with the RDP accountant at delta 1e-5). The DP
families have 8 models each (about 8 minutes per model), so every attack has 7 references.

| Training | Held-out accuracy | Loss AUC | Best TPR at 0.1% FPR | Best TPR at 1% FPR |
|---|---:|---:|---:|---:|
| Non-private (64 models) | 93.0% | 54.6% | 1.49% (LiRA online) | 6.66% (LiRA online) |
| DP-SGD, epsilon 8.00 (sigma 0.718) | 67.9% | 55.7% | 0.13% (shadow) | 1.29% (LiRA offline) |
| DP-SGD, epsilon 1.00 (sigma 2.396) | 13.0% | 51.2% | 0.11% (loss, shadow) | 1.09% (loss) |

At epsilon 8 the loss attack still separates members (AUC 55.7%, from the 76.3% train vs 67.9%
held-out accuracy gap), but no attack beats chance at 0.1% FPR: the best is 0.13% against 0.10%
for a random guess. The (epsilon, delta) bound on any attack at 0.1% FPR is vacuous at epsilon 8
(99.97%), so the measured leakage is far below what the guarantee allows. With 7 references,
LiRA's per-example variance is unreliable (online AUC 48.5%), and the authors' scoring code returns
no score for examples without both IN and OUT references. At epsilon 1 every attack is at chance
(AUC 47.1% to 51.3%, TPR at 0.1% FPR at most 0.11%, against a bound of 0.27%), but the model is
no longer useful: held-out accuracy falls to 13.0% (train 13.9%) on 100 classes with this MLP and
30 epochs. Raw results: [`mia_purchase100-dp8.json`](benchmarks/results/mia_purchase100-dp8.json),
[`mia_purchase100-dp1.json`](benchmarks/results/mia_purchase100-dp1.json).

### Privacy and utility of DP-SGD on Purchase100

The cost of the guarantee: the same MLP trained once per target epsilon on a seeded 80% of
Purchase100 (157,859 records) and tested on the other 39,465, three seeds each, on the CPU. DP-SGD
uses Poisson sampling (expected batch 1,024, 4,620 steps over 30 epochs), clipping norm 1 and lr
1.0; sigma is calibrated with the RDP accountant at delta 1e-5, and the PRV and GDP epsilons of
the same schedule are recorded next to it. "Clip only" clips per-example gradients without noise,
which separates the cost of clipping from the cost of noise.

| Training | sigma | epsilon (RDP / PRV / GDP) | Test accuracy | Train accuracy |
|---|---:|---:|---:|---:|
| Non-private | | | 95.0% (0.1) | 99.4% |
| Clip only | 0 | | 92.0% (0.2) | 95.9% |
| DP-SGD | 0.659 | 8.00 / 7.08 / 6.07 | 69.5% (0.2) | 76.3% |
| DP-SGD | 0.841 | 4.00 / 3.60 / 3.28 | 61.3% (0.1) | 67.3% |
| DP-SGD | 1.188 | 2.00 / 1.84 / 1.76 | 43.4% (0.2) | 47.8% |
| DP-SGD | 1.943 | 1.00 / 0.92 / 0.90 | 22.5% (0.7) | 24.2% |

Mean over seeds 0, 1 and 2 (standard deviation in parentheses). Clipping alone costs 3.0 points;
the noise costs the rest, 22.5 points at epsilon 8 and 69.5 points at epsilon 1. The PRV
accountant certifies the same noise at an epsilon 8% to 12% lower than RDP (GDP is an asymptotic
approximation, not an upper bound). Reproduce: `python benchmarks/dp_utility.py purchase100 --eps 1
2 4 8 --seeds 0 1 2 --device cpu`; raw results:
[`dp_utility_purchase100.jsonl`](benchmarks/results/dp_utility_purchase100.jsonl).

<p align="center">
  <img src="docs/figures/dp_utility.png" alt="Test accuracy of DP-SGD against epsilon on Purchase100" width="560">
  <img src="docs/figures/dp_attack.png" alt="LiRA TPR at low FPR under DP-SGD against the bound implied by epsilon" width="560">
</p>

### Canaries and extraction on Pythia-70M (WikiText-2)

Pythia-70M (deduplicated) is fine-tuned for 4 epochs (2,260 steps, 18.9M trainable parameters in
the block linear layers, 95 minutes on MPS) on a seeded half of the WikiText-2 training paragraphs
(8,778, max 128 tokens) plus 24 fictional canaries "The vault code of &lt;name&gt; is d d d d d.", four each
inserted 1, 2, 4, 8, 16 and 32 times (9,030 training sequences). Exposure is exact: all 100,000
five-digit secrets are scored, so the rank of the true secret is known (maximum log2 100,000 =
16.61 bits); "extracted" means greedy decoding of the prefix returns the secret.

| Inserted | Base: mean rank | Base: exposure | Fine-tuned: ranks | Fine-tuned: exposure | Skew-normal estimate | Extracted |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 59,754 | 1.43 | 23, 17, 80, 13 | 11.95 | 11.08 | 0 of 4 |
| 2 | 51,400 | 1.32 | 2, 1, 1, 1 | 16.36 | 14.77 | 3 of 4 |
| 4 | 55,529 | 1.94 | 1, 1, 1, 1 | 16.61 | 14.81 | 4 of 4 |
| 8 | 48,235 | 1.35 | 1, 1, 1, 1 | 16.61 | 15.04 | 4 of 4 |
| 16 | 50,367 | 1.02 | 1, 1, 1, 1 | 16.61 | 16.52 | 4 of 4 |
| 32 | 51,070 | 1.01 | 1, 1, 1, 1 | 16.61 | 16.61 | 4 of 4 |

A single insertion already lifts the secret from a random rank to the top 100 of 100,000; two
insertions make it extractable. The skew-normal estimate of Carlini et al. (2019) misses by up
to 5.1 bits on a single canary (it understates the top-ranked secrets and overstates some others
by up to 0.7 bits), which is why the exact count is reported. Sequence membership inference on 4,000 member and 4,000 non-member paragraphs (500
bootstrap resamples, 95% intervals):

| Attack on the fine-tuned model | AUC | TPR at 0.1% FPR | TPR at 1% FPR |
|---|---:|---:|---:|
| Loss | 79.1% [78.1, 80.1] | 0.63% [0.02, 2.25] | 9.35% [7.80, 10.99] |
| Min-k% probability (k = 20%) | 77.3% [76.3, 78.3] | 0.90% [0.08, 1.27] | 6.95% [5.69, 8.91] |
| zlib ratio | 70.9% [69.8, 72.0] | 1.98% [0.97, 3.03] | 7.55% [6.13, 8.98] |
| Reference (base model loss subtracted) | 75.8% [74.7, 76.9] | 0.73% [0.10, 1.79] | 6.25% [4.70, 7.83] |
| Loss on the base model (control) | 50.1% [48.8, 51.2] | 0.22% [0.07, 0.47] | 1.23% [0.60, 1.58] |

Fine-tuning lowers validation perplexity from 112.5 to 76.6. Untargeted extraction found nothing:
of 1,000 unconditional top-k samples of 96 tokens, none contains a 32-token window that appears
verbatim in the member or the non-member paragraphs, for the base or the fine-tuned model, also
among the top 100 ranked by perplexity, reference ratio or zlib ratio. The control row confirms
the split carries no signal before fine-tuning. The base model and fine-tuning ran on MPS; the
fine-tuned model was evaluated on the CPU (20 minutes, 1.4 GB peak) after MPS ran out of memory.
Reproduce: `python benchmarks/lm_canaries.py --models base ft --device cpu`; raw results:
[`lm_canaries.json`](benchmarks/results/lm_canaries.json).

### Comparison with references

| Check | Reference | Result | Verdict |
|---|---|---|---|
| LiRA online scores | Authors' released scoring (`lira-online-ref`), same logits | AUC 74.7% vs 74.8% (CIFAR-10), 65.1% vs 65.2% (Purchase100); TPR at 0.1% FPR 3.27% vs 3.27%, 1.30% vs 1.32% | tie |
| LiRA offline scores | Authors' offline scoring (two-sided) | 66.0% vs 57.6% AUC on CIFAR-10; the one-sided test here is stronger | win |
| CIFAR-10 LiRA online, TPR at 0.1% FPR | Carlini et al. 2022, Table I: 8.4% (WRN, 256 shadow models, augmented queries; quoted, not rerun) | 4.05% with a 0.11M CNN, 63 references, 2 queries | loss (smaller model and fewer references; the scaling curve is still rising at 63) |
| Purchase100 LiRA online, TPR at 0.1% FPR | Carlini et al. 2022, Table VIII: 4.1% (quoted, not rerun) | 1.49% with a regularised MLP at 93.0% held-out accuracy | loss (less overfit target) |
| RDP epsilon, 48 settings | Opacus 1.6 `RDPAccountant` | max relative difference 4.2e-12 (Rust), 1.6e-12 (NumPy) | tie |
| PRV epsilon, 41 settings | Opacus 1.6 `PRVAccountant` | max relative difference 2.1e-8, median 3.9e-10 | tie |
| Noise calibration, 12 settings | Opacus `get_noise_multiplier` | max relative sigma difference 3.2e-5 (Opacus stops at an epsilon tolerance of 1e-4) | tie |
| ROC AUC, 100,000 scores | scikit-learn `roc_curve` + `auc` | absolute difference 1.1e-16 | tie |

### Speed of the Rust core

Median wall time per call, `python benchmarks/parity_speed.py`
([`parity_speed.json`](benchmarks/results/parity_speed.json)).

| Operation | Rust core | NumPy reference | Library |
|---|---:|---:|---:|
| RDP epsilon (q 0.004, sigma 1.1, 7,500 steps) | **0.81 ms** | 12.1 ms | 22.7 ms (Opacus) |
| Noise calibration to epsilon 3 | **18 ms** | 778 ms | |
| ROC summary, 100,000 scores | **21 ms** | 54 ms | 60 ms (scikit-learn) |
| Bootstrap, 1,000 replicates of 100,000 scores | **7.9 s** | 32.6 s | |

### Number of reference models

TPR at 0.1% FPR of LiRA online (fixed variance) as the attacker trains more reference models (8
targets each): CIFAR-10 0.56% (2), 1.27% (8), 2.47% (16), 3.45% (32), 3.92% (63); Purchase100
0.49% (2), 0.88% (8), 1.15% (16), 1.37% (32), 1.50% (63).

<p align="center">
  <img src="docs/figures/lira_scaling.png" alt="LiRA TPR at low FPR against the number of reference models" width="860">
</p>

Full method, ablations and threats to validity: [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md).

## Quickstart

```bash
git clone https://github.com/adeoluo/privacy-attacks-lab && cd privacy-attacks-lab
uv sync --all-extras --dev --group validation   # builds the Rust core with maturin
uv run python -m privacy_attacks_lab epsilon --q 0.004 --sigma 1.1 --epochs 30
make test
```

Requires Python 3.12 or newer (tested on 3.12 to 3.15), a Rust toolchain and [uv](https://docs.astral.sh/uv/). Without the
Rust core the package falls back to the NumPy implementations.

## Usage

### Command line

```bash
python -m privacy_attacks_lab epsilon --q 0.004 --sigma 1.1 --epochs 30 --accountant prv
python -m privacy_attacks_lab calibrate --q 0.004 --epochs 30 --target-epsilon 1
python -m privacy_attacks_lab max-tpr --epsilon 1            # TPR bound implied by (eps, delta)-DP
python -m privacy_attacks_lab report benchmarks/results/mia_cifar10.json --sort tpr@0.001
python -m privacy_attacks_lab serve --port 8000
```

### Library

```python
from privacy_attacks_lab.accountants import epsilon, calibrate_noise
from privacy_attacks_lab import metrics

eps = epsilon(q=256 / 50_000, sigma=1.1, steps=5_860, delta=1e-5, accountant="prv")
sigma = calibrate_noise(8.0, 1e-5, q=1_024 / 98_662, steps=2_880)
summary = metrics.summarize(scores, is_member)  # AUC, balanced accuracy, TPR at low FPR
intervals = metrics.bootstrap(scores, is_member, reps=1_000)
```

Training families and attacking them:

```bash
scripts/download_data.sh                                  # CIFAR-10, Purchase100, Texas100, WikiText-2
python benchmarks/train_shadows.py purchase100            # 64 models into $PAL_DATA_DIR/runs/
python benchmarks/mia_benchmark.py purchase100            # attack table, ROC, bootstrap, scaling
python benchmarks/dp_utility.py purchase100 --eps 1 2 4 8 --seeds 0 1 2 --device cpu
python benchmarks/lm_canaries.py --models base ft --device cpu  # canary exposure, extraction
python benchmarks/figures.py                              # figures from committed results
```

### HTTP service

`GET /health`, `GET /ready`, `GET /metrics` (Prometheus text), and JSON `POST /epsilon`,
`/calibrate`, `/max-tpr`, `/roc` and `/exposure`. Standard library server, 8 MB body limit,
binds to 127.0.0.1 by default.

```bash
curl -s localhost:8000/epsilon -d '{"q": 0.004, "sigma": 1.1, "steps": 7500}'
curl -s localhost:8000/roc -d '{"scores": [0.9, 0.2, 0.7, 0.1], "labels": [1, 0, 1, 0]}'
```

### Demos

Recorded from real runs with [vhs](https://github.com/charmbracelet/vhs); each tape is in `docs/demo/`.

Privacy accounting: epsilon of a DP-SGD schedule, the noise for a target epsilon, and the attack TPR that epsilon allows.

<p align="center">
  <img src="docs/demo/accounting.gif" alt="Privacy accounting from the command line" width="860">
</p>

Attack tables from the committed CIFAR-10 and Purchase100 results, sorted by TPR at 0.1% FPR.

<p align="center">
  <img src="docs/demo/report.gif" alt="Membership inference attack tables for CIFAR-10 and Purchase100" width="860">
</p>

The HTTP service: readiness, epsilon from three accountants, canary exposure, and request metrics.

<p align="center">
  <img src="docs/demo/http.gif" alt="The HTTP service answering accounting and exposure requests" width="860">
</p>

## Design decisions

- **One family of 64 models, leave one out** instead of separate shadow and target models: every
  model is a target once, so 16 targets cost no extra training ([ADR-0002](docs/adr/0002-lira-leave-one-out-model-family.md)).
- **DP-SGD with per-sample gradients from hooks, Poisson sampling, GroupNorm** instead of shuffled
  batches or BatchNorm, so the training matches the accountant's assumptions ([ADR-0003](docs/adr/0003-dpsgd-per-sample-gradients.md)).
- **RDP in Rust, PRV in NumPy, both validated against Opacus**; RDP and ROC are hot loops in
  calibration and bootstrap, PRV is FFT bound ([ADR-0004](docs/adr/0004-accountants-rdp-prv-rust.md)).
- **Exact canary exposure by enumeration** on Pythia-70M rather than the skew-normal estimate
  alone, so the estimator's error is measured ([ADR-0005](docs/adr/0005-exact-canary-exposure.md)).
- **Fixed-variance LiRA reported next to per-example variance**: with 63 references the fixed
  variance is stronger at 0.1% FPR on both datasets, matching the paper's advice for few models.

## Testing

```bash
make test        # unit, property-based, integration (CLI and HTTP)
make coverage    # with the 80% gate used in CI
make parity      # Rust core against the NumPy reference
make validate    # accountants against Opacus, ROC against scikit-learn
make rust-check  # cargo fmt, clippy and the Rust unit tests
```

348 Python tests (98% line coverage in CI) and 13 Rust tests: property-based tests with
hypothesis (accountant monotonicity, ROC invariants, exposure bounds), Opacus parity of RDP, PRV
and calibration, Rust against NumPy parity, DP-SGD per-sample gradients against explicit
per-example backward passes and against Opacus, LiRA against the authors' scoring, CLI and HTTP
integration tests.
CI runs on Python 3.12, 3.13, 3.14 and 3.15 with small fixtures; full-scale runs are the benchmark scripts.

## Responsible AI

- **Model card:** [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)
- **Data card:** [`docs/DATA_CARD.md`](docs/DATA_CARD.md)
- **Threat model:** [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md)
- **Standards:** the [threat model](docs/THREAT_MODEL.md#standards-mapping) maps the attacks to
  LLM02:2026 Sensitive Information Disclosure in the OWASP Top 10 for LLM Applications 2026 and
  to MITRE ATLAS v2026.10 (AML.T0137.000 Training Data Information from AI Models: Infer
  Membership, AML.T0137.003 Training Data Information from AI Models: Extract Memorized Data,
  and the defence AML.M0041 Privacy-Preserving AI Training and Inference).

The attacks only target models this project trains on public benchmark data; the canaries are
fictional names and random digits. Attack success is an empirical lower bound on leakage, never a
proof of privacy; only the DP-SGD guarantee is a bound.

## Limitations

- Models are small (0.11M and 0.48M parameters) to train 64 of each on an 8 GB M1; larger models
  memorise more, so the TPRs here understate what LiRA achieves on WRN-sized models.
- 64 models per dataset, not the paper's 256; the reference-count curve is still rising at 63.
- Texas100 is downloaded and supported by the loader and presets but was not benchmarked.
- The language model was fine-tuned without DP only. The script has DP-SGD presets (epsilon 8
  and 1, ghost clipping) but each would take several hours on this machine and was not run.
- Untargeted extraction used 1,000 samples; the extraction literature uses hundreds of thousands,
  so zero verbatim hits does not show that nothing is extractable.
- The attacker knows the exact training recipe and data pool, a strong and favourable setting.
- Wall-clock timings were taken on a machine shared with other jobs and vary between runs.

## Roadmap

- [ ] Texas100 family of 64 models.
- [ ] 256 reference models on CIFAR-10 with a WRN-16-4, as in the LiRA paper.
- [ ] DP-SGD fine-tuning of the canary language model at epsilon 8 and 1.
- [ ] Auditing DP-SGD with canary gradients (lower bounds on epsilon from attacks).

## License

Released under the [MIT License](LICENSE).
