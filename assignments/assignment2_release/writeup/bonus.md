# Bonus: Swin-T backbone and COCO

The required detector is FCOS with a ResNet-18 backbone. These two extensions leave that loss and the box decoder alone. They change the network that produces the feature maps, and the dataset those maps are trained on.

Fill in the tables after the 4070 runs. The curves are the TensorBoard events under `logs/`.

## Swin Transformer backbone

### What changed

`FCOSBackbone` in `code/det_libs/modeling.py` builds either a ResNet or Swin-T. The config is `code/configs/det_voc_fcos_swin.yaml`.

Swin-T is a hierarchical vision transformer. A 4x4 patch embed turns the image into a map of stride 4. Three patch-merging layers then halve the resolution and double the channels, so the four stages sit at strides 4, 8, 16, and 32 with 96, 192, 384, and 768 channels. That is the same idea as a ResNet stem plus layer1–4, except each stage is windowed self-attention instead of a residual convolution.

The rest of this FCOS already assumes five levels at strides 8, 16, 32, 64, and 128. ResNet supplies the first three from `layer2`, `layer3`, and `layer4`, and the FPN builds the last two by strided convolution. Swin is wired the same way:

| Pyramid level | Swin module | Stride | Channels |
| --- | --- | --- | --- |
| P3 | `features.3` | 8 | 192 |
| P4 | `features.5` | 16 | 384 |
| P5 | `features.7` | 32 | 768 |
| P6, P7 | FPN, unchanged | 64, 128 | 256 after the FPN |

`features.1` (stride 4) is not used. Adding it would mean a new regression range and a new point-generator stride. Keeping the pyramid the same means the center-sampling loss and `FCOS.inference` do not need a second code path.

### Why the maps are permuted

Torchvision's Swin attends over the last dimension, so a stage tensor is `(batch, height, width, channels)`. The FPN convolutions expect `(batch, channels, height, width)`. `FCOSBackbone.forward` permutes Swin outputs and calls `contiguous()` so the convolution does not see a tensor that is stored in the other order. ResNet outputs are already in the convolution layout, so they are returned as-is.

### Why these training choices

- ImageNet initialization (`Swin_T_Weights.IMAGENET1K_V1`), same reason the ResNet-18 backbone is not trained from scratch. VOC is small; a random Swin will not learn windows and boxes in 10 epochs.
- AdamW at `1e-4`, weight decay `0.05`, instead of the ResNet SGD at `5e-3`. Swin was pretrained with AdamW. SGD at the ResNet learning rate tends to wreck a pretrained transformer in the first epoch.
- Weight decay skips biases and `relative_position_bias_table`. That table is one learned scalar per pair of positions inside a 7x7 window. It is an embedding, not a weight matrix. `build_optimizer` used to ignore it and then raise, because every parameter must sit in exactly one of the two groups. The new clause puts the table in the no-decay group.
- `backbone_freeze_bn: false`. Swin has LayerNorm, not BatchNorm. LayerNorm uses the tokens inside one sample, so a batch of 2 does not make its statistics noisy. The freeze helper only matches `BatchNorm2d`, so leaving the flag on would have done nothing. It is off so the config says what actually happens.
- Batch size 2 and the same 256–320 short side as the ResNet VOC config. Swin keeps an attention matrix per window and the deepest map is 768 channels wide. That is the setting to try first on an 8GB 4070. Increase the batch to 4 only if the process does not run out of memory.
- 10 epochs plus 1 warmup, matching the required VOC schedule, so the Swin curve and the ResNet-18 curve cover the same number of passes.

### How to run

From `assignment2_release/code`, after VOC is downloaded:

```bash
python ./train_det.py ./configs/det_voc_fcos_swin.yaml --output swin_t
python ./eval_det.py ./configs/det_voc_fcos_swin.yaml ../logs/det_voc_fcos_swin_swin_t/model_final.pth.tar -v
```

### Results

| | ResNet-18 (required config) | Swin-T |
| --- | --- | --- |
| mAP@0.5 | 58.3% (already measured) | fill in |
| Train time | about 25 min on the 4070 | fill in |
| Checkpoint | `logs/det_voc_fcos_local4070` | `logs/det_voc_fcos_swin_swin_t` |

Paste the TensorBoard loss curve here after training.

## COCO 2017

### What changed

`COCODetection` in `code/det_libs/dataset.py`, the config `code/configs/det_coco_fcos.yaml`, and `download_coco.sh`.

`build_dataset` used to know only VOC. It now also accepts `name: COCO` with splits `train2017` and `val2017`. VOC images all live in one directory and the json is `trainval.json` / `test.json`. COCO images live in `train2017/` and `val2017/`, and the json files are `instances_train2017.json` and `instances_val2017.json`. The loader checks both filename patterns and, when a split subdirectory exists, reads images from there. VOC's folder has no `trainval/` subdirectory, so the old path still resolves.

### Why category ids are remapped

`ConvertAnnotations` does `category_id - 1` and the head has `num_classes` channels. That is correct for VOC, whose ids are 1 through 20 with no gaps. It is wrong for COCO. COCO detection has 80 classes, but the ids run from 1 to 90 and skip 12 numbers (there is no category 12, for example). Subtracting 1 from those ids would ask the head for a channel it does not have, and pycocotools would look up the wrong class.

`COCODetection` sorts the categories by id and renames them to 1..80 before the existing transform subtracts 1. The model still trains on dense labels 0..79. At evaluation, `to_coco_category_id` maps the 1-based prediction back to the original COCO id before it is written into the results json. VOC's version of that function returns the label unchanged, so the VOC numbers do not move. `train_det.py` also checks that `model.num_classes` equals the number of categories in the json and stops if a COCO config is left at 20.

### Why the training set is 8000 images

train2017 has about 118,000 images. Four epochs of the full set on a laptop is most of a day, and the assignment says a subset is enough to show the implementation works. `max_images: 8000` draws a permutation with seed 0 and keeps the first 8000 indexes. The seed is fixed so a resumed checkpoint sees the same images. The validation loader does not use `max_images`, so mAP is on all 5,000 val2017 images. Set `max_images: null` in the yaml to train on every training image.

The backbone stays ResNet-18 with the VOC optimizer (SGD, learning rate 0.005, batch 4, short side 256–320, one warmup epoch and four cosine epochs). The thing being demonstrated is the data path, not a new architecture. A 4-epoch subset will not match a published COCO number. The loss should drop, and val mAP should be well above a detector that emits nothing.

### How to run

From `assignment2_release`:

```bash
bash download_coco.sh
```

That downloads annotations, val2017 (about 1 GB), and train2017 (about 18 GB). Then, from `code`:

```bash
python ./train_det.py ./configs/det_coco_fcos.yaml --output res18_8k
python ./eval_det.py ./configs/det_coco_fcos.yaml ../logs/det_coco_fcos_res18_8k/model_final.pth.tar -v
```

Report both the COCO summary (AP at IoU 0.50:0.95, and AP at IoU 0.50) and the wall time printed by `eval_det.py`.

### Results

| | Value |
| --- | --- |
| Train images | 8000 / ~118k, seed 0 |
| Epochs | 1 warmup + 4 |
| AP @ IoU 0.50:0.95 | fill in |
| AP @ IoU 0.50 | fill in |
| Eval time on val2017 | fill in |

Paste the training-loss curve here after the run.
