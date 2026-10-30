import math
import torch
import torchvision

from torchvision.models import resnet
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from torchvision.ops.boxes import batched_nms

import torch
from torch import nn
from torch.nn import functional as F

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss


class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 2.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=3, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, x):
        """
        The head will be applied to all levels of the feature pyramid, and predict 
        a single logit for each location on every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will the featuer map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x C x H x W, storing the classification logits (scores).

        """
        outputs = tuple()
        for feats in x:
            out = self.conv(feats)
            out = self.cls_logits(out)
            outputs += (out, )
        return outputs


class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 2.
    """

    def __init__(self, in_channels, num_convs=3):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        The logic is rather similar to FCOSClassificationHead. The key difference is 
        that this head bundles both regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        """
        reg_outputs = tuple()
        ctr_outputs = tuple()
        for feats in x:
            out = self.conv(feats)
            reg_out = self.bbox_reg(out)
            ctr_out = self.bbox_ctrness(out)
            reg_outputs += (reg_out, )
            ctr_outputs += (ctr_out, )
        return reg_outputs, ctr_outputs


class SwinStageExtractor(nn.Module):
    """Swin-T stages as a normal Module, not a traced graph.

    create_feature_extractor (used for ResNet below) traces the network.
    On Swin that trace replaces each attention Linear with a plain Module
    that still owns a `weight` tensor but is no longer nn.Linear.
    build_optimizer then cannot tell those matrices from an unrecognized
    parameter and refuses to build. Walking `features` ourselves keeps the
    real Linear, LayerNorm, and relative-position table, which is also what
    you see if you print a block while debugging.

    `stage_names` looks like ["features.3", "features.5", "features.7"].
    The number is the index into Swin-T's `features` Sequential:
    0 patch embed, 1 stage, 2 merge, 3 stage, 4 merge, 5 stage, 6 merge, 7 stage.
    """

    def __init__(self, stage_names):
        super().__init__()
        # The classification head on this model is not registered here.
        # Detection does not use it, and leaving it out keeps those
        # parameters out of the optimizer.
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.features = model.features
        self.stage_names = list(stage_names)
        self.stage_indexes = [self._index(name) for name in self.stage_names]

    @staticmethod
    def _index(name):
        prefix = "features."
        suffix = name[len(prefix):] if name.startswith(prefix) else ""
        if not suffix.isdigit():
            raise ValueError(
                "Swin backbone_out_feats entries must look like 'features.3', got %r."
                % name
            )
        return int(suffix)

    def forward(self, x):
        # Collect by index, then emit in the yaml's order. The FPN pairs
        # backbone_out_feats_dims[i] with the i-th entry of this dict, so
        # the order has to follow the config, not the order we happened to
        # compute the stages in.
        found = {}
        wanted = set(self.stage_indexes)
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in wanted:
                found[index] = x
        missing = [name for name, index in zip(self.stage_names, self.stage_indexes) if index not in found]
        if missing:
            raise RuntimeError(
                "Swin-T features has %d modules; these stages were not produced: %s"
                % (len(self.features), missing)
            )
        return {
            name: found[index]
            for name, index in zip(self.stage_names, self.stage_indexes)
        }


class FCOSBackbone(nn.Module):
    """ImageNet backbone that always returns NCHW feature maps for the FPN.

    ResNet stages are already NCHW: channel is dimension 1, which is what
    FeaturePyramidNetwork's convolutions read. Torchvision's Swin-T is not.
    Its blocks attend over the last dimension, so each stage comes out as
    NHWC (batch, height, width, channel). This module permutes those maps
    once, here, so the FPN, the heads, and the loss stay the same object
    for both backbones.

    Which stages are returned is chosen by the yaml (`backbone_out_feats`).
    For Swin-T the maps that line up with ResNet layer2/3/4 are:

        features.3  stride 8    192 channels
        features.5  stride 16   384 channels
        features.7  stride 32   768 channels

    features.1 (stride 4, 96 channels) is left out of the FPN on purpose.
    It still runs, because features.3 is built from it. The FPN then adds
    stride 64 and 128 from the stride-32 map. The point generator, the
    regression ranges, and the loss already assume strides 8, 16, 32, 64, 128.
    """

    RESNETS = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")

    def __init__(self, name, return_nodes):
        super().__init__()
        self.channels_last = False
        if name in self.RESNETS:
            # Same initialization the provided ResNet-18 detector uses.
            # ImageNet weights give the FPN something that already sees edges
            # and textures; training from scratch on VOC does not.
            backbone_model = resnet.__dict__[name](weights="IMAGENET1K_V1")
            self.extractor = create_feature_extractor(
                backbone_model, return_nodes=return_nodes
            )
        elif name == "swin_t":
            # Smallest Swin (about 28M parameters), which is the practical
            # choice on an 8GB laptop GPU. IMAGENET1K_V1 is the classification
            # checkpoint; detection fine-tunes it.
            self.extractor = SwinStageExtractor(list(return_nodes))
            self.channels_last = True
        else:
            raise ValueError(
                "Unsupported backbone %r. Use one of %s or 'swin_t'."
                % (name, ", ".join(self.RESNETS))
            )

    def forward(self, x):
        features = self.extractor(x)
        if not self.channels_last:
            return features
        # permute is a view with a different stride. contiguous() makes the
        # following convolutions happy; some of them refuse a non-contiguous
        # NCHW tensor that is really stored as NHWC.
        return {
            name: value.permute(0, 3, 1, 2).contiguous()
            for name, value in features.items()
        }


class FCOS(nn.Module):
    """
    Implementation of Fully Convolutional One-Stage (FOCS) object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network. ResNet-18/34/50/101/152 or swin_t.
        backbone_freeze_bn (bool): if to freeze batch norm in the backbone
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to be defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_freeze_bn,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
    ):
        super().__init__()
        if len(backbone_out_feats) != len(backbone_out_feats_dims):
            raise ValueError(
                "backbone_out_feats and backbone_out_feats_dims must list the "
                "same stages. Got %s and %s."
                % (list(backbone_out_feats), list(backbone_out_feats_dims))
            )
        # The classification and regression towers use GroupNorm(16, C).
        # C has to be divisible by 16 or that layer cannot be built.
        if fpn_feats_dim % 16 != 0:
            raise ValueError(
                "fpn_feats_dim must be divisible by 16 (GroupNorm groups). Got %d."
                % fpn_feats_dim
            )
        self.backbone_name = backbone
        self.backbone_freeze_bn = backbone_freeze_bn
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # ResNet or Swin. Both return a dict of NCHW maps keyed by the names
        # in backbone_out_feats. Swin has no BatchNorm; backbone_freeze_bn
        # only switches BatchNorm2d to eval, so it is a no-op for Swin.
        self.backbone = FCOSBackbone(backbone, return_nodes)

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
            extra_blocks=LastLevelP6P7(fpn_feats_dim, fpn_feats_dim)
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = 100
        self.loss_normalizer_momentum = 0.9

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        # fix all bn ops, yet still allow the update of affine params
        if self.backbone_freeze_bn:
            self.apply(self.freeze_bn)
        return self

    """
    The behavior of the forward function changes depending on if the model is
    in training or evaluation mode.

    During training, the model expects both the input tensors
    (list of tensors within the range of [0, 1]), as well as a targets
    (list of dictionary), containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - other keys such as image_id are not used here
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        # sanity check
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features)

        # training / inference
        if self.training:
            # training: generate GT labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            # return detection results during inference
            return detections

    @staticmethod
    def _boxes_from_ltrb(points_yx, ltrb):
        """Decode (y, x) points and (l, t, r, b) distances into xyxy boxes.

        PointGenerator builds its grid with meshgrid(indexing="ij") and stacks
        [row, col], so channel 0 is the vertical coordinate. Swapping these
        two channels silently mirrors every box and collapses mAP.
        l, t, r, b are distances in the same pixel units as the points:
        x1 = x - l, y1 = y - t, x2 = x + r, y2 = y + b.
        """
        y = points_yx[..., 0]
        x = points_yx[..., 1]
        left, top, right, bottom = ltrb.unbind(dim=-1)
        return torch.stack((x - left, y - top, x + right, y + bottom), dim=-1)

    def _flatten_heads(self, cls_logits, reg_outputs, ctr_logits, points, strides, reg_range):
        """Concat pyramid levels into (N, P, ...) predictions and (P, ...) locations.

        Heads arrive as NCHW. Permute to NHWC before flattening so location i
        lines up with points.reshape(-1, 2), which is also row-major. Levels
        stay in PointGenerator order (stride 8 first). The regression range is
        repeated once per location so the assignment can compare max(l, t, r, b)
        against that level's [min, max] without another Python loop.
        """
        cls_levels = []
        reg_levels = []
        ctr_levels = []
        point_levels = []
        stride_levels = []
        range_levels = []
        for cls_level, reg_level, ctr_level, pts, stride, level_range in zip(
            cls_logits, reg_outputs, ctr_logits, points, strides, reg_range
        ):
            batch_size, _, height, width = cls_level.shape
            num_locations = height * width
            cls_levels.append(cls_level.permute(0, 2, 3, 1).reshape(batch_size, num_locations, -1))
            reg_levels.append(reg_level.permute(0, 2, 3, 1).reshape(batch_size, num_locations, 4))
            ctr_levels.append(ctr_level.permute(0, 2, 3, 1).reshape(batch_size, num_locations))
            flat_points = pts.reshape(num_locations, 2)
            point_levels.append(flat_points)
            stride_levels.append(flat_points.new_full((num_locations,), float(stride)))
            range_levels.append(
                level_range.to(device=flat_points.device, dtype=flat_points.dtype)
                .view(1, 2)
                .expand(num_locations, 2)
            )
        return (
            torch.cat(cls_levels, dim=1),
            torch.cat(reg_levels, dim=1),
            torch.cat(ctr_levels, dim=1),
            torch.cat(point_levels, dim=0),
            torch.cat(stride_levels, dim=0),
            torch.cat(range_levels, dim=0),
        )

    def _assign_one_image(self, gt_boxes, gt_labels, points, strides, reg_ranges):
        """Center-sampled FCOS targets for one image (journal FCOS, TPAMI 2020).

        Vectorized over locations P and ground-truth boxes G. The only Python
        loops in the loss sit outside this function (batch and FPN levels).

        A location is positive for a box only when all three hold:
          1. It is strictly inside the box (every ltrb distance > 0).
          2. It is inside the center region: the box center expanded by
             center_sampling_radius * stride on each side, then intersected
             with the box. Default radius is 1.5, so only cells near the
             center of a small object become positives.
          3. max(l, t, r, b), in pixels, falls inside this level's
             regression_range (inclusive). That is what assigns large objects
             to coarse levels and small objects to fine ones.

        If several boxes claim one location, the smallest area wins. Everywhere
        else is background. Regression targets stored for the network are the
        pixel distances divided by the level stride, matching the head, which
        is trained to predict distances in stride units.
        """
        num_locations = points.shape[0]
        device = points.device
        cls_target = points.new_zeros((num_locations, self.num_classes))
        reg_target = points.new_zeros((num_locations, 4))
        ctr_target = points.new_zeros((num_locations,))
        pos_mask = torch.zeros(num_locations, dtype=torch.bool, device=device)
        if gt_boxes.numel() == 0:
            return cls_target, reg_target, ctr_target, pos_mask

        y = points[:, 0]
        x = points[:, 1]
        x1, y1, x2, y2 = gt_boxes.unbind(dim=-1)
        # Distances from each location to each box, in pixels. Shapes are (P, G).
        left = x[:, None] - x1[None, :]
        top = y[:, None] - y1[None, :]
        right = x2[None, :] - x[:, None]
        bottom = y2[None, :] - y[:, None]
        ltrb = torch.stack((left, top, right, bottom), dim=-1)

        # (P, G): a location is inside a box when all four distances are positive.
        inside_box = ltrb.min(dim=-1).values > 0
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        # radius is per location because each pyramid level has its own stride.
        radius = strides[:, None] * self.center_sampling_radius
        # Intersect the expanded center square with the ground-truth box so a
        # location just outside the object is never labeled positive.
        center_x1 = torch.maximum(center_x[None, :] - radius, x1[None, :])
        center_y1 = torch.maximum(center_y[None, :] - radius, y1[None, :])
        center_x2 = torch.minimum(center_x[None, :] + radius, x2[None, :])
        center_y2 = torch.minimum(center_y[None, :] + radius, y2[None, :])
        inside_center = (
            (x[:, None] > center_x1)
            & (y[:, None] > center_y1)
            & (x[:, None] < center_x2)
            & (y[:, None] < center_y2)
        )
        # Range limits are in pixels, so compare the raw distances, not ltrb/stride.
        max_offset = ltrb.max(dim=-1).values
        inside_range = (max_offset >= reg_ranges[:, None, 0]) & (
            max_offset <= reg_ranges[:, None, 1]
        )
        is_positive = inside_box & inside_center & inside_range

        # Ambiguous locations take the smallest box. Non-matches are +inf so
        # min() ignores them; a row that stays inf is background.
        areas = ((x2 - x1) * (y2 - y1)).clamp(min=0)
        match_area = areas[None, :].expand(num_locations, -1).masked_fill(~is_positive, float("inf"))
        min_area, min_index = match_area.min(dim=1)
        pos_mask = torch.isfinite(min_area)
        location_index = torch.arange(num_locations, device=device)
        chosen_ltrb = ltrb[location_index, min_index]
        # Eq. 1: network targets are distances divided by the feature stride.
        reg_target = chosen_ltrb / strides[:, None].clamp(min=1e-6)
        reg_target = torch.where(pos_mask[:, None], reg_target, torch.zeros_like(reg_target))

        # Labels from the dataloader are already 0-based (category_id - 1).
        # Background stays the all-zero row, which focal loss treats as negative.
        chosen_labels = gt_labels[min_index]
        if pos_mask.any():
            cls_target[pos_mask] = F.one_hot(
                chosen_labels[pos_mask].long(), num_classes=self.num_classes
            ).to(dtype=cls_target.dtype)

        # Centerness uses only the winning box. Locations that lost the match
        # can have a negative distance on the chosen index, and a negative
        # ratio under the square root is NaN. Clamp before the ratio, then
        # zero the target wherever pos_mask is false.
        left_c, top_c, right_c, bottom_c = chosen_ltrb.unbind(dim=-1)
        left_c = left_c.clamp(min=0)
        top_c = top_c.clamp(min=0)
        right_c = right_c.clamp(min=0)
        bottom_c = bottom_c.clamp(min=0)
        centerness = torch.sqrt(
            (torch.minimum(left_c, right_c) / torch.maximum(left_c, right_c).clamp(min=1e-6))
            * (torch.minimum(top_c, bottom_c) / torch.maximum(top_c, bottom_c).clamp(min=1e-6))
        )
        ctr_target = torch.where(pos_mask, centerness, torch.zeros_like(centerness))
        return cls_target, reg_target, ctr_target, pos_mask

#########################################################################################
# Part II-2: FCOS training
#########################################################################################
    def compute_loss(
        self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
    ):
        """Focal classification, GIoU regression, and centerness BCE.

        All three terms are sums divided by an exponential moving average of
        the number of positive locations. The EMA starts at 100 (set in
        __init__) with momentum 0.9, and the divisor is clamped to at least 1
        so an all-background batch does not divide by zero. final_loss is the
        sum the trainer back-propagates.
        """
        cls_preds, reg_preds, ctr_preds, flat_points, flat_strides, flat_ranges = (
            self._flatten_heads(cls_logits, reg_outputs, ctr_logits, points, strides, reg_range)
        )
        batch_size = cls_preds.shape[0]
        cls_targets = []
        reg_targets = []
        ctr_targets = []
        pos_masks = []
        for image_index in range(batch_size):
            gt_boxes = targets[image_index]["boxes"]
            gt_labels = targets[image_index]["labels"]
            cls_t, reg_t, ctr_t, pos = self._assign_one_image(
                gt_boxes, gt_labels, flat_points, flat_strides, flat_ranges
            )
            cls_targets.append(cls_t)
            reg_targets.append(reg_t)
            ctr_targets.append(ctr_t)
            pos_masks.append(pos)
        cls_targets = torch.stack(cls_targets, dim=0)
        reg_targets = torch.stack(reg_targets, dim=0)
        ctr_targets = torch.stack(ctr_targets, dim=0)
        pos_mask = torch.stack(pos_masks, dim=0)

        num_pos = int(pos_mask.sum().item())
        # Feed the EMA at least 1 so a batch with no positives does not
        # drag the running foreground count all the way to zero.
        num_pos_for_ema = float(max(num_pos, 1))
        self.loss_normalizer = (
            self.loss_normalizer_momentum * self.loss_normalizer
            + (1.0 - self.loss_normalizer_momentum) * num_pos_for_ema
        )
        normalizer = max(self.loss_normalizer, 1.0)

        # Focal loss (alpha 0.25, gamma 2) is summed over every location and
        # class, including background. It is the only term that sees negatives.
        cls_loss = sigmoid_focal_loss(cls_preds, cls_targets, reduction="sum") / normalizer
        if num_pos > 0:
            # The head predicts ltrb / stride. GIoU is not invariant to that
            # scaling because the box corner is point +/- distance, and the
            # point itself is in pixels. Multiply by the stride before decoding.
            pred_boxes = self._boxes_from_ltrb(
                flat_points, reg_preds * flat_strides.view(1, -1, 1)
            )
            target_boxes = self._boxes_from_ltrb(
                flat_points, reg_targets * flat_strides.view(1, -1, 1)
            )
            reg_loss = giou_loss(
                pred_boxes[pos_mask], target_boxes[pos_mask], reduction="sum"
            ) / normalizer
            ctr_loss = F.binary_cross_entropy_with_logits(
                ctr_preds[pos_mask], ctr_targets[pos_mask], reduction="sum"
            ) / normalizer
        else:
            # An image with no ground truth still has to produce a grad-connected
            # zero. A Python float 0 would give the regression head a None grad.
            reg_loss = reg_preds.sum() * 0.0
            ctr_loss = ctr_preds.sum() * 0.0

        losses = {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "ctr_loss": ctr_loss,
            "final_loss": cls_loss + reg_loss + ctr_loss,
        }
        return losses


#########################################################################################
# Part II-1: FCOS Inference
#########################################################################################
    def _candidates_one_level(self, cls_level, reg_level, ctr_level, points, stride, image_hw):
        """Scores, decoded boxes, and class ids for one image at one pyramid level.

        The objectness used for thresholding and NMS is
        sigmoid(class logit) * sigmoid(centerness), broadcast over classes.
        Flattening locations and classes together means one cell can emit more
        than one category, and topk_candidates counts those class-location
        pairs rather than cells. Regression range is not applied here; every
        level is decoded and NMS sorts them out.

        Returned labels are still the 0-based head index. inference() adds 1
        so they match the COCO category ids the evaluator expects.
        """
        image_h, image_w = image_hw
        # (C, H, W) -> (H * W, C). Permute before reshape so row-major order
        # matches points. Distances come out of the head in stride units.
        class_probs = torch.sigmoid(cls_level)
        centerness = torch.sigmoid(ctr_level)
        scores = (class_probs * centerness).permute(1, 2, 0).reshape(-1, self.num_classes)
        distances = reg_level.permute(1, 2, 0).reshape(-1, 4) * float(stride)
        flat_points = points.reshape(-1, 2)
        y = flat_points[:, 0]
        x = flat_points[:, 1]
        # The transform pads the batch to a common size. image_shapes is the
        # unpadded resized (h, w), so a center past that edge is padding and
        # should not become a detection. Boxes are clipped to the same box.
        inside = (y >= 0) & (x >= 0) & (y < image_h) & (x < image_w)
        scores = scores[inside]
        distances = distances[inside]
        flat_points = flat_points[inside]
        if scores.numel() == 0:
            empty_boxes = cls_level.new_zeros((0, 4))
            empty_scores = cls_level.new_zeros((0,))
            empty_labels = torch.zeros(0, dtype=torch.long, device=cls_level.device)
            return empty_boxes, empty_scores, empty_labels

        # Threshold first, then keep the top scores at this level. Index
        # layout is location-major: index // num_classes is the cell and
        # index % num_classes is the class.
        scores_flat = scores.reshape(-1)
        candidate_index = torch.where(scores_flat > self.score_thresh)[0]
        candidate_scores = scores_flat[candidate_index]
        if candidate_scores.numel() > self.topk_candidates:
            topk_scores, topk_index = torch.topk(candidate_scores, self.topk_candidates)
            candidate_index = candidate_index[topk_index]
            candidate_scores = topk_scores
        if candidate_index.numel() == 0:
            empty_boxes = cls_level.new_zeros((0, 4))
            empty_scores = cls_level.new_zeros((0,))
            empty_labels = torch.zeros(0, dtype=torch.long, device=cls_level.device)
            return empty_boxes, empty_scores, empty_labels

        location_index = torch.div(candidate_index, self.num_classes, rounding_mode="floor")
        class_index = candidate_index % self.num_classes
        boxes = self._boxes_from_ltrb(flat_points[location_index], distances[location_index])
        boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=image_w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=image_h)
        # Drop boxes that clipping reduced to a sliver. A side under one
        # pixel is not a usable detection and can destabilize NMS.
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        keep = (widths >= 1.0) & (heights >= 1.0)
        return boxes[keep], candidate_scores[keep], class_index[keep]

    def inference(
        self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes
    ):
        """Per image: decode every FPN level, class-wise NMS, shift labels by +1.

        NMS is per class because batched_nms groups boxes by the label tensor.
        The +1 happens after NMS so suppression still groups the 0-based head
        indices, and the dict handed back uses the 1-20 ids COCOeval expects.
        An image with nothing above the score threshold returns empty tensors
        rather than a missing key.
        """
        num_images = cls_logits[0].shape[0]
        detections = []
        for image_index in range(num_images):
            image_hw = image_shapes[image_index]
            level_boxes = []
            level_scores = []
            level_labels = []
            for level_index, (cls_level, reg_level, ctr_level, level_points) in enumerate(
                zip(cls_logits, reg_outputs, ctr_logits, points)
            ):
                boxes, scores, labels = self._candidates_one_level(
                    cls_level[image_index],
                    reg_level[image_index],
                    ctr_level[image_index],
                    level_points,
                    strides[level_index],
                    image_hw,
                )
                level_boxes.append(boxes)
                level_scores.append(scores)
                level_labels.append(labels)
            boxes = torch.cat(level_boxes, dim=0)
            scores = torch.cat(level_scores, dim=0)
            labels = torch.cat(level_labels, dim=0)
            if boxes.numel() > 0:
                # batched_nms suppresses overlaps independently inside each class.
                keep = batched_nms(boxes, scores, labels, self.nms_thresh)
                keep = keep[: self.detections_per_img]
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep] + 1
            else:
                labels = labels.to(dtype=torch.long)
            detections.append({"boxes": boxes, "scores": scores, "labels": labels})
        return detections