# ADR-0004: RDP and PRV accountants in Python, hot loops in a Rust core

- **Status:** Accepted
- **Date:** 2026-11-04

## Context
Every privacy/utility point needs the noise multiplier for a target epsilon, found by bisection over
an accountant, and every attack evaluation needs ROC curves and bootstrap intervals on up to 200,000
examples with 1,000 resamples. Accountants must agree with a trusted reference to many digits, since a
silent error in epsilon invalidates every DP claim.

## Decision
Implement the RDP accountant of the Poisson subsampled Gaussian (Mironov, Talwar and Zhang 2019) and
the PRV accountant (Gopi, Lee and Wutschitz 2021) in numpy and scipy, plus the Gaussian DP CLT
accountant as a cross-check. Validate against Opacus' accountants (opt-in `validation` test group).
Move the hot loops (the RDP series and ROC/bootstrap sorting) into a Rust core compiled with PyO3 and
maturin as `privacy_attacks_lab.pal_rs`, with the numpy code kept as the parity reference.

## Alternatives considered
- **Call Opacus' accountants** - correct, but the project's purpose is an independent check.
- **Google's `dp_accounting`** - another strong reference, but a heavier dependency for CI.
- **numpy only** - simpler build, but the bootstrap dominated evaluation time.

## Consequences
- Parity tests compare Rust and numpy results; CI builds the Rust core on every push.
- PRV and RDP answers differ by design (PRV is tighter); both are reported side by side.
