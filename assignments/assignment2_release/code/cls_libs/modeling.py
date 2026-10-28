import numpy as np
import torch
import torch.nn as nn
import math

from .blocks import (PatchEmbed, window_partition, window_unpartition,
                    DropPath, MLP, trunc_normal_)


################################################################################
# You will need to fill in the missing code in this file
################################################################################


################################################################################
# Part I.1: A simple convolutional neural network
################################################################################
class SimpleNet(nn.Module):
    # a simple CNN for image classifcation
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(SimpleNet, self).__init__()
        # you can start from here and create a better model
        self.features = nn.Sequential(
            # conv1 block: conv 7x7
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv2 block: simple bottleneck
            conv_op(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(64, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv3 block: simple bottleneck
            conv_op(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(128, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def reset_parameters(self):
        # init all params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
################################################################################
# Part I.2: Design your own CNN
################################################################################
class ResidualBlock(nn.Module):
    """Basic residual block used by ResidualNet (He et al., post-activation).

    Two 3x3 convolutions, each followed by BatchNorm. The first conv carries
    the stride when the stage downsamples. Conv bias is off because the
    following BatchNorm already has a learnable shift. The skip is the identity
    when shape is unchanged, and a 1x1 projection otherwise, so the add is
    always valid. ReLU is applied after the skip is added.
    """

    def __init__(self, in_channels, out_channels, stride=1, conv_op=nn.Conv2d):
        super().__init__()
        self.conv1 = conv_op(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv_op(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Projection only when spatial size or channel count changes.
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv_op(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        # No ReLU between the second BN and the skip: the residual path
        # must be able to pass a signed correction.
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResidualNet(nn.Module):
    """From-scratch scene CNN: 3x3 stem and four residual stages.

    SimpleNet has no normalization and no skip path, so it is hard to train
    once the stack gets deeper. This net adds both. Channel schedule is
    64-128-256-512 with two basic blocks per stage. The stem already halves
    the 128 crop, so stage 0 keeps stride 1 and later stages stride by 2.
    Global average pooling then a linear layer produces the 100 MiniPlaces
    logits. No ImageNet weights: every parameter is learned on the train split.
    """

    def __init__(self, conv_op=nn.Conv2d, num_classes=100, blocks_per_stage=(2, 2, 2, 2)):
        super().__init__()
        self.conv_op = conv_op
        # Stride-2 stem. Bias is off because BatchNorm follows immediately.
        self.stem = nn.Sequential(
            conv_op(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        channels = (64, 128, 256, 512)
        stages = []
        in_channels = 64
        for stage_idx, out_channels in enumerate(channels):
            # Stage 0 stays at the stem resolution; each later stage halves it.
            stride = 1 if stage_idx == 0 else 2
            blocks = [
                ResidualBlock(in_channels, out_channels, stride=stride, conv_op=conv_op)
            ]
            for _ in range(1, blocks_per_stage[stage_idx]):
                blocks.append(
                    ResidualBlock(out_channels, out_channels, stride=1, conv_op=conv_op)
                )
            stages.append(nn.Sequential(*blocks))
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        # Same scheme as SimpleNet.reset_parameters, plus the linear head.
        # Kaiming fan_out matches ReLU convs; BN scale starts at 1 so the
        # first forward pass is not randomly rescaled.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# change this to your model!
default_cnn_model = ResidualNet

################################################################################
# Part I.3: Understanding self-attention and Transformer block
################################################################################
class Attention(nn.Module):
    """Multi-head Self-Attention."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
    ):
        """
        Args:
            dim (int): Number of input channels. We assume Q, K, V will be of
                same dimension as the input.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # linear projection for query, key, value
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # linear projection at the end
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # input size (B, H, W, C)
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(
                B, H * W, 3, self.num_heads, -1
            ).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, head_dim).
        # Heads are folded into the batch so one matmul runs all of them.
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        # Scaled dot-product attention. self.scale is head_dim ** -0.5,
        # which keeps the softmax from saturating as the head grows.
        # Softmax is over keys (last axis), so each query sums to 1.
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        # Merge heads: (B * nHead, HW, head_dim) -> (B, nHead, HW, head_dim)
        # -> (B, HW, nHead, head_dim) -> (B, H, W, C), then the output proj.
        out = out.reshape(B, self.num_heads, H * W, -1)
        out = out.permute(0, 2, 1, 3).reshape(B, H, W, -1)
        out = self.proj(out)
        return out

class TransformerBlock(nn.Module):
    """Transformer blocks with support of local window self-attention"""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        window_size=0,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            window_size (int): Window size for window attention blocks.
                If it equals 0, then global attention is used.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.drop_path = DropPath(drop_path) if drop_path>0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer
        )

        self.window_size = window_size

    def forward(self, x):
        # Pre-LN: normalize first, then attend. The residual adds back the
        # un-normalized shortcut, not the tensor that entered attention.
        shortcut = x
        x = self.norm1(x)

        # window_size == 0 attends over the whole feature map.
        # Otherwise the map is padded and split into non-overlapping windows.
        # Each window is one (B, window, window, C) sequence, which is the
        # layout Attention.forward already expects, so the same module serves
        # both global and local attention. Unpartition strips the pad and
        # restores the original H, W.
        if self.window_size > 0:
            height, width = x.shape[1], x.shape[2]
            windows, pad_hw = window_partition(x, self.window_size)
            windows = self.attn(windows)
            x = window_unpartition(windows, self.window_size, pad_hw, (height, width))
        else:
            x = self.attn(x)

        # Two residual branches. DropPath is nn.Identity when the rate is 0,
        # so stochastic depth is a no-op unless a block was given a nonzero rate.
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


#################################################################################
# Part I.4: Design and train a vision Transformer
#################################################################################
class SimpleViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=128,
        num_classes=100,
        patch_size=16,
        in_chans=3,
        embed_dim=192,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        window_size=4,
        window_block_indexes=(0, 2),
    ):
        """
        Args:
            img_size (int): Input image size.
            num_classes (int): Number of object categories
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            window_size (int): Window size for local attention blocks.
            window_block_indexes (list): Indexes for blocks using local attention.
                Local window attention allows more efficient computation, and
                can be coupled with standard global attention. E.g., [0, 2]
                indicates the first and the third blocks will use local window
                attention, while other block use standard attention.

        Feel free to modify the default parameters here.
        """
        super(SimpleViT, self).__init__()
        assert (num_heads > 0) and (embed_dim % num_heads == 0) 

        if use_abs_pos:
            # Initialize absolute positional embedding with image size
            # The embedding is learned from data
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # patch embedding layer
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Alternate local and global attention. Only indexes listed in
        # window_block_indexes (default 0 and 2) receive window_size; every
        # other block gets 0, which TransformerBlock treats as global attention.
        # dpr increases with depth, so later blocks drop more residual branches.
        blocks = []
        for block_idx in range(depth):
            block_window = window_size if block_idx in window_block_indexes else 0
            blocks.append(
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[block_idx],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    window_size=block_window,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        # Final norm and the classifier are created before apply(_init_weights)
        # so the linear head is initialized with the rest of the linears.
        # There is no class token: the head reads a mean over patch tokens.
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)
        # add any necessary weight initialization here

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # PatchEmbed is a strided conv followed by BCHW -> BHWC, so tokens
        # stay on a 2D grid. pos_embed has that same (1, H, W, C) shape and
        # is added only when use_abs_pos built it.
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        # Mean over the spatial axes is the classification token. Output is
        # (B, num_classes) logits for the cross-entropy trainer.
        x = self.norm(x)
        x = x.mean(dim=(1, 2))
        x = self.head(x)
        return x

# change this to your model!
default_vit_model = SimpleViT



