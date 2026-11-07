# Data Card - Privacy Attacks Lab

## Sources and provenance
| Dataset | Source | Records | Features | Classes | Raw size |
|---|---|---:|---|---:|---:|
| CIFAR-10 | Krizhevsky (2009), https://www.cs.toronto.edu/~kriz/cifar.html | 60,000 (50,000 train, 10,000 test) | 32x32x3 images | 10 | 163 MB |
| Purchase100 | Shokri et al. (S&P 2017), privacytrustlab/datasets | 197,324 | 600 binary | 100 | 21 MB |
| Texas100 | Shokri et al. (S&P 2017), privacytrustlab/datasets | 67,330 | 6,169 binary | 100 | 15 MB |
| WikiText-2 (raw) | Merity et al. (2016), https://wikitext.smerity.com | 17,556 train, 1,841 validation paragraphs | text | n/a | 4.5 MB |

`scripts/download_data.sh` downloads every archive with a pinned SHA-256 into
`$PAL_DATA_DIR` (default `../.bench-data/privacy-attacks-lab`, outside the repository). Nothing is
redistributed in the repository; CI uses small synthetic fixtures.

## Licensing
- CIFAR-10: released by the authors for research use.
- Purchase100 and Texas100: derived by Shokri et al. from the Kaggle "Acquire Valued Shoppers"
  challenge and the Texas Department of State Health Services hospital discharge public use data;
  used here as published research benchmarks, unmodified.
- WikiText-2: Creative Commons Attribution-ShareAlike (from Wikipedia).
- Pythia-70M (deduplicated), the language model: Apache-2.0 (EleutherAI).

## Composition and splits
- CIFAR-10: the 50,000 training images are the pool of the membership experiments; each model of a
  family trains on a random half (25,000) and the 10,000 test images measure accuracy.
- Purchase100: all 197,324 records are the pool; each model trains on a random half (98,662) and its
  other half is its test set. DP utility runs use a seeded 80/20 split (157,859 / 39,465).
- Texas100: all 67,330 records; each model trains on 33,665.
- WikiText-2: a seeded half of the training paragraphs (plus inserted canaries) fine-tunes the
  language model (members); the other half never trains it (non-members); the validation
  paragraphs measure perplexity.

## Preprocessing
- CIFAR-10 is read with a restricted unpickler that only resolves the numpy constructors the archive
  needs, then stored as uint8 in `cifar10.npz`; images are scaled to [0, 1] and normalised per channel.
- Purchase100 and Texas100 are parsed line by line into `npz` files (Texas100 bit packed, 52 MB
  instead of 415 MB).
- WikiText-2: headings and empty lines are dropped; paragraphs are tokenised with the Pythia
  tokenizer and truncated to 128 tokens.

## Canaries (synthetic)
24 fictional sentences of the form "the vault code of <fictional name> is d d d d d", with random
five-digit secrets, inserted 1, 2, 4, 8, 16 or 32 times (four canaries per level). Names are invented;
no real person, account or secret is used.

## Known biases and gaps
- Purchase100 and Texas100 have heavy class imbalance and were built for membership inference
  research; Texas100 contains hospital records that were de-identified by the publisher.
- Results on these datasets describe these models and training recipes, not deployed systems.

## Ethical considerations
The attacks are run only against models trained here, on public benchmark data. The canary secrets
are synthetic. The point of the work is to measure leakage so that the defence (DP-SGD) can be
calibrated, not to extract data from third-party models.
