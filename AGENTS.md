# AGENTS.md

Working notes for coding agents and contributors in **Privacy Attacks Lab**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

Membership inference attacks (LiRA, shadow models, calibrated loss), DP-SGD with RDP and PRV accountants checked against Opacus, and canary exposure and extraction from small language models.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `rust/`: Rust core compiled into a Python extension with PyO3
- `scripts/`: maintenance scripts (release, assets, data)
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `rust/src/lib.rs`: Rust core of privacy-attacks-lab.
- `rust/src/python.rs`: PyO3 bindings, built by maturin into the `privacy_attacks_lab.pal_rs` module.
- `rust/src/rdp.rs`: Renyi DP of the Poisson subsampled Gaussian mechanism, its conversion to (epsilon, delta) and noise calibration.
- `rust/src/roc.rs`: ROC analysis of membership inference scores and its bootstrap.
- `src/privacy_attacks_lab/__init__.py`: Privacy Attacks Lab - package root.
- `src/privacy_attacks_lab/accountants/__init__.py`: Privacy accountants for DP-SGD (the Poisson subsampled Gaussian mechanism).
- `src/privacy_attacks_lab/accountants/gdp.py`: Gaussian differential privacy (GDP) accounting by the central limit theorem.
- `src/privacy_attacks_lab/accountants/prv.py`: Privacy random variable (PRV) accountant for the Poisson subsampled Gaussian mechanism.
- `src/privacy_attacks_lab/accountants/rdp.py`: Renyi differential privacy (RDP) of the Poisson subsampled Gaussian mechanism.
- `src/privacy_attacks_lab/attacks/__init__.py`: Membership inference attacks.
- `src/privacy_attacks_lab/attacks/calibrated.py`: Difficulty-calibrated loss attack (Watson, Guo, Cormode and Sablayrolles, ICLR 2022).
- `src/privacy_attacks_lab/attacks/lira.py`: The Likelihood Ratio Attack, LiRA (Carlini, Chien, Nasr, Song, Terzis and Tramer, S&P 2022).
- `src/privacy_attacks_lab/attacks/lira_reference.py`: LiRA scoring exactly as in the authors' released code, kept as a parity reference.
- `src/privacy_attacks_lab/attacks/loss.py`: The loss-threshold attack (Yeom, Giacomelli, Fredrikson and Jha, CSF 2018).
- `src/privacy_attacks_lab/attacks/shadow.py`: The shadow-model attack (Shokri, Stronati, Song and Shmatikov, S&P 2017).
- `src/privacy_attacks_lab/canary.py`: Canaries and exposure (Carlini, Liu, Erlingsson, Kos and Song, USENIX Security 2019).
- `src/privacy_attacks_lab/cli.py`: Command line interface: ``python -m privacy_attacks_lab <command>``.
- `src/privacy_attacks_lab/data.py`: Datasets: CIFAR-10, Purchase100 and Texas100, converted once to compact ``.npz`` files.
- `src/privacy_attacks_lab/dpsgd.py`: DP-SGD (Abadi et al., CCS 2016), implemented directly on ``torch.func``.
- `src/privacy_attacks_lab/evaluation.py`: Leave-one-out evaluation of membership inference attacks on a family of models.
- `src/privacy_attacks_lab/lm.py`: A small causal language model fine-tuned on text with inserted canaries.
- `src/privacy_attacks_lab/metrics.py`: ROC metrics for membership inference: AUC, balanced accuracy and TPR at low FPR.
- `src/privacy_attacks_lab/models.py`: Small target and shadow models.
- `src/privacy_attacks_lab/service.py`: HTTP service: privacy accounting, attack metrics and canary exposure over JSON.
- `src/privacy_attacks_lab/shadow.py`: Train many models on random halves of a data pool and store their per-example signals.
- `src/privacy_attacks_lab/signals.py`: Per-example signals that membership inference attacks read off a model's logits.
- `src/privacy_attacks_lab/training.py`: Training and inference loops for target, shadow and reference models.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make rust-check` | Rust core: fmt, clippy (with and without Python), unit tests |
| `make rust-develop` | Rebuild the Rust core into .venv after editing rust/ (uv sync also builds it) |
| `make parity` | Rust vs Python parity tests |
| `make validate` | Validation against Opacus and scikit-learn |
| `make run` | Start the HTTP service on port 8000 |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut with `make cut-release`.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.

## Rust notes

- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` must pass.
- No `unwrap()` in library code paths that can fail at runtime; return a typed error.
- Unsafe code is limited to the places that need a syscall and carries a `SAFETY:` comment.
