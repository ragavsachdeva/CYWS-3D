import kornia as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class UpSamplingBlock(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self._conv_bn_layer1 = nn.Sequential(
            create_conv_layer_with_kaiming_init(input_dims, hidden_dims, kernel_size=3, stride=1, padding=1),
            create_batch_norm_layer_with_custom_init(hidden_dims),
        )
        self._conv_bn_layer2 = nn.Sequential(
            create_conv_layer_with_kaiming_init(hidden_dims, output_dims, kernel_size=3, stride=1, padding=1),
            create_batch_norm_layer_with_custom_init(output_dims),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=True)
        x = self._conv_bn_layer1(x)
        x = F.gelu(x)
        x = self._conv_bn_layer2(x)
        x = F.gelu(x)
        return x


class DimReductionBlock(nn.Module):
    def __init__(self, list_of_input_output_tuples, rearrange_before=None, rearrange_after=None):
        super().__init__()
        layers = []
        if rearrange_before is not None:
            layers.append(rearrange_before)
        for input_dim, output_dim in list_of_input_output_tuples:
            layers.append(create_linear_layer_with_xavier_init(input_dim, output_dim))
            layers.append(nn.GELU())
        if rearrange_after is not None:
            layers.append(rearrange_after)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            create_conv_layer_with_kaiming_init(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            create_conv_layer_with_kaiming_init(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(create_conv_layer_with_kaiming_init(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Sequence2SpatialBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.keep_cls = args.encoder.keep_cls_in_seq2spatial
        h = 1 + ((224 - args.encoder.patch) // args.encoder.stride)
        w = 1 + ((224 - args.encoder.patch) // args.encoder.stride)
        self.seq_to_spatial = Rearrange("b (h w) c -> b h w c", h=h, w=w)
        self.norm = nn.LayerNorm(args.encoder.output_dim)

    def forward(self, x):
        x = self.norm(x)
        tokens = self.seq_to_spatial(x[:, 1:])
        if self.keep_cls:
            cls = repeat(x[:, 0], "b c -> b h w c", h=tokens.shape[1], w=tokens.shape[2])
            tokens = rearrange([tokens, cls], "two b h w c -> b h w (two c)")
        return rearrange(tokens, "b h w c -> b c h w")


class FeatureFusionBlock(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, output_resolution):
        super().__init__()
        self.out_res = output_resolution
        self._conv_bn_layer1 = nn.Sequential(
            create_conv_layer_with_kaiming_init(input_dims, hidden_dims, kernel_size=3, stride=1, padding=1),
            create_batch_norm_layer_with_custom_init(hidden_dims),
        )
        self._conv_bn_layer2 = nn.Sequential(
            create_conv_layer_with_kaiming_init(hidden_dims, output_dims, kernel_size=3, stride=1, padding=1),
            create_batch_norm_layer_with_custom_init(output_dims),
        )

    def forward(self, x, y):
        bicubic_resize = K.augmentation.Resize(y.shape[-2:], resample=2, keepdim=True)
        x = bicubic_resize(x)
        xy = torch.cat([x, y], dim=1)
        xy = self._conv_bn_layer1(xy)
        xy = F.gelu(xy)
        xy = self._conv_bn_layer2(xy)
        xy = F.gelu(xy)
        bicubic_resize = K.augmentation.Resize(self.out_res, resample=2, keepdim=True)
        return bicubic_resize(xy)


class ResidualConvBock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = create_conv_layer_with_kaiming_init(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = create_conv_layer_with_kaiming_init(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        out = F.gelu(x)
        out = self.conv1(out)
        out = F.gelu(out)
        out = self.conv2(out)
        return x + out


class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        if downsample:
            self.conv1 = create_conv_layer_with_kaiming_init(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                create_conv_layer_with_kaiming_init(in_channels, out_channels, kernel_size=1, stride=2),
                create_batch_norm_layer_with_custom_init(out_channels),
            )
        else:
            self.conv1 = create_conv_layer_with_kaiming_init(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential(
                create_conv_layer_with_kaiming_init(in_channels, out_channels, kernel_size=1, stride=1),
                create_batch_norm_layer_with_custom_init(out_channels),
            )

        self.conv2 = create_conv_layer_with_kaiming_init(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = create_batch_norm_layer_with_custom_init(out_channels)
        self.bn2 = create_batch_norm_layer_with_custom_init(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)


def create_conv_layer_with_kaiming_init(*args, **kwargs):
    conv_layer = nn.Conv2d(*args, **kwargs)
    nn.init.kaiming_normal_(conv_layer.weight, mode="fan_out", nonlinearity="relu")
    return conv_layer


def create_batch_norm_layer_with_custom_init(*args, **kwargs):
    batch_norm_layer = nn.BatchNorm2d(*args, **kwargs)
    nn.init.constant_(batch_norm_layer.weight, 1)
    nn.init.constant_(batch_norm_layer.bias, 0)
    return batch_norm_layer


def create_linear_layer_with_xavier_init(*args, **kwargs):
    linear_layer = nn.Linear(*args, **kwargs)
    nn.init.xavier_uniform_(linear_layer.weight)
    return linear_layer
