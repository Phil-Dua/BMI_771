# FCOS on PASCAL VOC 2007

This is the trained model from the assignment default configuration, not the pretrained checkpoint. The pretrained model is a separate inference check and is expected to reach about 60.9% mAP at IoU 0.5.

## Test result

Evaluated `logs/det_voc_fcos_local4070/model_final.pth.tar` with `code/configs/det_voc_fcos.yaml` on the VOC 2007 test set (4,952 images).

The assignment metric is mAP at IoU 0.5:

- **mAP@0.5 = 58.3%** (COCO `AP @[ IoU=0.50 ] = 0.583`)
- Inference time: **157.67 seconds**

Other COCO summary numbers from the same run, for the table in the report:

| Quantity | Value |
|---|---|
| AP, IoU 0.50:0.95, all | 0.317 |
| AP, IoU 0.50, all | 0.583 |
| AP, IoU 0.75, all | 0.310 |
| AP, IoU 0.50:0.95, small | 0.048 |
| AP, IoU 0.50:0.95, medium | 0.171 |
| AP, IoU 0.50:0.95, large | 0.404 |
| AR, IoU 0.50:0.95, max 1 det | 0.311 |
| AR, IoU 0.50:0.95, max 10 dets | 0.453 |
| AR, IoU 0.50:0.95, max 100 dets | 0.466 |
| AR, IoU 0.50:0.95, small | 0.103 |
| AR, IoU 0.50:0.95, medium | 0.331 |
| AR, IoU 0.50:0.95, large | 0.563 |

Raw detections are in `logs/det_voc_fcos_local4070/eval_results.json`.

## Training

Command: `python ./train_det.py ./configs/det_voc_fcos.yaml --output local4070`

Settings are the assignment defaults: ResNet-18 backbone with ImageNet initialization, frozen backbone batch norm, FPN width 256, batch size 4, SGD with learning rate 0.005, momentum 0.9, weight decay 0.0001, cosine decay, 1 warmup epoch and 10 training epochs. Short side is jittered over 256, 288, and 320 pixels. Long side is capped at 480. Mixed precision (fp16).

Training took about 25 minutes on a local RTX 4070 Laptop GPU. Average loss by epoch:

| Epoch | Average loss |
|---|---|
| 0 | 1.82 |
| 1 | 1.58 |
| 2 | 1.34 |
| 3 | 1.32 |
| 4 | 1.27 |
| 5 | 1.23 |
| 6 | 1.21 |
| 7 | 1.12 |
| 8 | 1.13 |
| 9 | 1.07 |
| 10 | 1.11 |

Final-epoch components: classification 0.22, box regression 0.26, centerness 0.63.

Checkpoint and TensorBoard logs: `logs/det_voc_fcos_local4070/`. The file to submit is `model_final.pth.tar`.
