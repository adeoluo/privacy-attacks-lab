# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.2.0] 2026-11-07

### Features

- Add membership inference results against DP-SGD on Purchase100 (#12)
- Add DP-SGD privacy/utility curves on Purchase100 (#15)
- Add canary exposure and extraction results on Pythia-70M (#16)

### Performance

- Bound the canary benchmark to five-digit secrets and release MPS memory (#10)
- Make the canary benchmark resumable and score sequences in batches of 8 (#11)
- Fine-tune the canary model in chunks of 4 to stay under the MPS memory cap (#13)
- Checkpoint canary model fine-tuning every epoch and halve the enumeration batch (#14)

### Maintenance

- Set the version to 0.2.0 and fix the citation metadata (#17)

## [0.1.0] 2026-11-07

### Features

- Add RDP, PRV and GDP accountants validated against Opacus (#1)
- Add membership inference attacks, ROC metrics and DP-SGD (#2)
- Add LiRA reference scoring, canary exposure and LM extraction (#3)
- Add HTTP service and attack report command (#4)
- Add Purchase100 results, DP-SGD families and figures (#6)
- Add accountant parity with Opacus and Rust core speed benchmark (#7)

### Documentation

- Add ADRs and the architecture diagram (#5)
- Write the README, evaluation report, figures and demos (#9)

### Maintenance

- Scaffold project
- Add release tooling, changelog and agent notes (#8)
