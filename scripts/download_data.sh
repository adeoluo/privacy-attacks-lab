#!/usr/bin/env bash
# Download the raw datasets (about 210 MB) into $PAL_DATA_DIR
# (default: ../.bench-data/privacy-attacks-lab next to the repository).
# Archives are converted to compact .npz files on first use by privacy_attacks_lab.data.load.
set -euo pipefail

DIR="${PAL_DATA_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/.bench-data/privacy-attacks-lab}"
mkdir -p "$DIR"

fetch() {
  local url="$1" name="$2" sha="$3"
  if [ ! -f "$DIR/$name" ]; then
    echo "downloading $name"
    curl -fL --retry 3 -o "$DIR/$name.part" "$url"
    mv "$DIR/$name.part" "$DIR/$name"
  fi
  echo "$sha  $DIR/$name" | shasum -a 256 -c -
}

# CIFAR-10, python version (Krizhevsky 2009): 170 MB.
fetch https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz cifar-10-python.tar.gz \
  6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce
# Purchase100 and Texas100 (Shokri et al. 2017), as distributed by privacytrustlab: 22 MB and 16 MB.
fetch https://github.com/privacytrustlab/datasets/raw/master/dataset_purchase.tgz dataset_purchase.tgz \
  5d11041c383e5885c14098a5043fd63c30f495408289815d3f257da58c4f3046
fetch https://github.com/privacytrustlab/datasets/raw/master/dataset_texas.tgz dataset_texas.tgz \
  a729757c8a00254342c7f9f8045ee039c5b404441997c52046a3152faf978839
