# ADR-0003: DP-SGD with three per-sample gradient engines, ghost clipping by default

- **Status:** Accepted
- **Date:** 2026-11-04

## Context
DP-SGD needs each example's gradient norm before clipping. Materialising every per-example gradient
costs `batch x parameters` memory, which on an 8 GB machine limits the logical batch (DP-SGD wants
batches of 1,024 or more). Opacus solves this with module hooks; the goal here is an implementation
that does not depend on Opacus but is checked against it.

## Decision
Implement DP-SGD in `dpsgd.py` with Poisson sampling, flat clipping and Gaussian noise divided by the
expected batch size, and three interchangeable engines for the per-example step:
`vmap` (`torch.func.vmap(grad(...))`, any model), `hooks` (one batched backward, per-example weight
gradients rebuilt from layer inputs and output gradients for Linear, Conv2d and GroupNorm, including
sequence inputs) and `ghost` (ghost clipping, Li et al. ICLR 2022: per-example norms of linear layers from
`||g_i||^2 ||a_i||^2`, or `<A A^T, G G^T>` on sequences, and the clipped sum as one matrix product
`(c * G)^T A`, so linear layers' per-example gradients are never materialised). `ghost` is the default. Physical chunks
are sized from the parameter count so at most about 256 MB of per-example gradients is held at once.

## Alternatives considered
- **Opacus `GradSampleModule`** - mature, but the point of the project is a readable
  implementation; Opacus is used only in the opt-in validation tests.
- **Microbatches of size 1** - simplest and exact, but 100 to 1,000 times slower.

## Consequences
- Tests check all engines against a per-example loop, ghost norms against materialised gradients,
  and per-example gradients against Opacus for every model architecture.
- New layer types need a hook rule; unsupported trainable layers raise instead of silently leaking.
- BatchNorm is not allowed (it mixes examples); the models use GroupNorm.
