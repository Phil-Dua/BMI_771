#!/bin/bash
# COCO 2017 images and instance annotations, in the layout det_coco_fcos.yaml expects.
#
# val2017 is about 1 GB. train2017 is about 18 GB. Annotations are about 250 MB.
# The training config uses 8000 train images, but those 8000 are a random
# subset of the full zip, so the zip is still required.
#
# Run from assignment2_release:
#   bash download_coco.sh

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/coco
cd data/coco

if [[ ! -f annotations/instances_train2017.json || ! -f annotations/instances_val2017.json ]]; then
  wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -n annotations_trainval2017.zip
fi

if [[ ! -d val2017 ]]; then
  wget -c http://images.cocodataset.org/zips/val2017.zip
  unzip -n val2017.zip
fi

if [[ ! -d train2017 ]]; then
  wget -c http://images.cocodataset.org/zips/train2017.zip
  unzip -n train2017.zip
fi

echo "COCO is ready under data/coco"
