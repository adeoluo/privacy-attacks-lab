# Evaluation Report: Privacy Attacks Lab

Every number here comes from a script in `benchmarks/` and a raw result file in
`benchmarks/results/`. Hardware for all runs: Apple M1 (8 cores, 8 GB unified memory, PyTorch MPS),
shared with other jobs, so wall-clock times are indicative only.

## 1. Membership inference

### Setup
- **Datasets.** CIFAR-10: the 50,000 training images form the pool (the 10,000 test images measure
  test accuracy). Purchase100: all 197,324 records form the pool. Texas100 is supported by the loader
  but not benchmarked.
- **Model families.** For each dataset, 64 models; model `i` trains on a seeded random half of the
  pool (mask `i`), so each example is IN for about 32 models and OUT for the rest
  ([ADR-0002](adr/0002-lira-leave-one-out-model-family.md)).
  - CIFAR-10: GroupNorm CNN, 114,186 parameters, SGD with momentum, lr 0.05, weight decay 5e-4,
    batch 256, 15 epochs, no augmentation. Train accuracy 98.5% (sd 1.0), held-out accuracy 75.5%
    (sd 0.3), test accuracy 75.2%. 90 s per model, 1.6 h in total.
  - Purchase100: MLP 600-512-256-128-100, 484,836 parameters, lr 0.05, weight decay 1e-5, batch
    256, 20 epochs. Train accuracy 99.8%, held-out accuracy 93.0% (sd 0.1). 32 s per model, 34 min.
- **Signals.** Every model's logits on every pool example (CIFAR-10: the image and its horizontal
  mirror). Attacks use the stable logit-scaled confidence `log p_y - log(1 - p_y)` (LiRA) or the
  cross-entropy loss (loss, calibrated).
- **Protocol.** Targets 0 to 15; for each target the remaining 63 models are the attacker's reference
  models. Metrics over all pool examples of the target: AUC, balanced accuracy, TPR at 0.1%, 1% and
  10% FPR. The README tables give the mean and standard deviation over the 16 targets; the result
  files also hold 1,000-replicate percentile bootstrap intervals for target 0.
- **Attacks.** Loss threshold (Yeom et al. 2018); difficulty calibrated loss, target loss minus the
  mean OUT loss of the references (Watson et al. 2022); per-class shadow attack models on sorted
  confidence vectors from 8 shadow models (Shokri et al. 2017); LiRA (Carlini et al. 2022) online
  (IN and OUT Gaussians) and offline (OUT only, one-sided), each with per-example or pooled
  ("fixed") variance; and the authors' released scoring code applied to the same logits as a check.

### Results (mean over 16 targets)

| Attack | CIFAR-10 AUC | CIFAR-10 TPR@0.1% | CIFAR-10 TPR@1% | Purchase100 AUC | Purchase100 TPR@0.1% | Purchase100 TPR@1% |
|---|---:|---:|---:|---:|---:|---:|
| Loss threshold | 56.0% | 0.10% | 1.02% | 54.6% | 0.10% | 1.00% |
| Calibrated loss | 64.5% | 1.75% | 6.10% | 62.4% | 1.13% | 4.69% |
| Shadow models | 71.9% | 0.99% | 5.68% | 58.1% | 0.14% | 1.34% |
| LiRA offline | 66.0% | 0.69% | 4.43% | 62.5% | 0.45% | 3.09% |
| LiRA offline, fixed variance | 66.1% | 0.91% | 4.79% | 63.5% | 1.02% | 5.03% |
| LiRA online | 74.7% | 3.27% | 10.71% | 65.1% | 1.30% | 5.87% |
| LiRA online, fixed variance | 74.9% | 4.05% | 12.04% | 65.5% | 1.49% | 6.66% |
| Authors' online scoring | 74.8% | 3.27% | 10.78% | 65.2% | 1.32% | 5.90% |
| Authors' online scoring, global variance | 74.0% | 3.62% | 11.19% | 65.2% | 1.36% | 6.28% |
| Authors' offline scoring (two-sided) | 57.6% | 0.46% | 3.02% | 51.7% | 0.10% | 1.39% |

Bootstrap 95% intervals for target 0, LiRA online with fixed variance: CIFAR-10 TPR at 0.1% FPR
4.61% [3.61, 5.16], AUC 75.5% [75.1, 75.9]; Purchase100 1.36% [1.10, 1.58], AUC 65.4% [65.2, 65.7].

### Ablations

- **Calibration is the largest single gain at low FPR.** Subtracting each example's average OUT loss
  moves TPR at 0.1% FPR from 0.10% to 1.75% (CIFAR-10) and 0.10% to 1.13% (Purchase100); the loss
  threshold is at chance at low FPR on both datasets.
- **Online beats offline.** IN models add 3.1 points of TPR at 0.1% FPR on CIFAR-10 and 0.5 points
  on Purchase100 (fixed variance).
- **Fixed variance beats per-example variance with 63 references** at 0.1% FPR on both datasets
  (CIFAR-10 4.05% vs 3.27%, Purchase100 1.49% vs 1.30%), as the paper reports for small model counts.
- **One-sided offline test.** Testing only for higher confidence than the OUT distribution gives
  66.0% AUC on CIFAR-10 against 57.6% for the two-sided test of the released code.
- **Reference models.** TPR at 0.1% FPR of LiRA online with fixed variance rises with the number of
  references (8 targets): CIFAR-10 0.56%, 0.81%, 1.27%, 2.47%, 3.45%, 3.92% at 2, 4, 8, 16, 32, 63;
  Purchase100 0.49%, 0.64%, 0.88%, 1.15%, 1.37%, 1.50%. Neither curve has flattened at 63.
- **Queries.** On CIFAR-10, using only the original image (no mirror) gives LiRA online with fixed
  variance 76.0% AUC and 4.75% TPR at 0.1% FPR, better than averaging the image and its mirror
  (74.9%, 4.05%): the models were trained without augmentation, so the mirror is a weaker signal.
  Result file: [`mia_cifar10_firstquery.json`](../benchmarks/results/mia_cifar10_firstquery.json).

### Comparison with published numbers (quoted, not rerun)
Carlini et al. (2022) report 8.4% TPR at 0.1% FPR on CIFAR-10 (Table I; a wide ResNet,
256 shadow models, 18 augmented queries) and 4.1% online and 1.4% offline on Purchase
(Table VIII). This lab measures 4.05% (CIFAR-10) and 1.49% (Purchase100) online, 0.91% and 1.02% offline. The gap on CIFAR-10
is consistent with the smaller model, 4x fewer references and 2 queries; on Purchase100 the target
here generalises well (93.0% held-out accuracy). The authors' scoring code applied to the same logits
matches this implementation, so the gap comes from the models, not the attack code.

## 2. Accountants and the Rust core

`python benchmarks/parity_speed.py` ([`parity_speed.json`](../benchmarks/results/parity_speed.json)).

- **RDP**: 48 settings, q in {0.001, 0.004, 0.01, 0.05}, sigma in {0.6, 1, 2, 4}, steps in
  {100, 1,000, 10,000}, delta 1e-5, epsilon from 0.10 to 186. Maximum relative difference from
  Opacus 1.6 `RDPAccountant`: 4.2e-12 (Rust), 1.6e-12 (NumPy).
- **PRV**: 41 of the settings (the 7 with the widest privacy loss grids skipped to bound memory),
  `eps_error` 0.01. Maximum relative difference from Opacus `PRVAccountant` 2.1e-8, median 3.9e-10.
- **Calibration**: 12 settings (target epsilon 1, 3, 8; q 0.004, 0.02; 10 or 30 epochs). Maximum
  relative sigma difference from `get_noise_multiplier` 3.2e-5, within Opacus' own epsilon tolerance.
- **Speed** (median per call): RDP epsilon 0.81 ms (Rust), 12.1 ms (NumPy), 22.7 ms (Opacus);
  calibration 18 ms vs 778 ms; ROC summary of 100,000 scores 21 ms (Rust), 54 ms (NumPy), 60 ms
  (scikit-learn `roc_curve` + `auc`); 1,000 bootstrap replicates 7.9 s vs 32.6 s.

## 3. Threats to validity
- **Small models.** The families are small enough to train 64 times on an 8 GB laptop. Attack
  success grows with memorisation, so these TPRs are not estimates for larger production models.
- **Strong attacker.** The attacker knows the training recipe and the pool and has 63 reference
  models; a weaker attacker does worse.
- **Targets are not independent.** The 16 targets share reference models, so the standard deviation
  across targets understates the uncertainty of the mean; per-target bootstrap intervals are given.
- **Shared hardware.** Timings vary with other load on the machine; the speed ratios were measured
  in the same process, one after the other.
- **Single training seed per mask.** Each model has one seed; variance from training randomness and
  from the data split are not separated.

## Reproducibility
```bash
scripts/download_data.sh
python benchmarks/train_shadows.py cifar10 && python benchmarks/mia_benchmark.py cifar10
python benchmarks/mia_benchmark.py cifar10 --queries first
python benchmarks/train_shadows.py purchase100 && python benchmarks/mia_benchmark.py purchase100
python benchmarks/parity_speed.py
python benchmarks/figures.py
```
Models and logits are written to `$PAL_DATA_DIR/runs/<preset>/` (default
`../.bench-data/privacy-attacks-lab`); the seeds are fixed in the presets.
