import math
import os
import pickle
import types
from typing import Tuple

import kornia as K
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.modules.utils as nn_utils
from easydict import EasyDict
from loguru import logger as L
from mmdet.models.dense_heads.centernet_head import CenterNetHead
from segmentation_models_pytorch.unet.model import UnetDecoder
from einops import rearrange

from modules.building_blocks import DownSamplingBlock, FeatureFusionBlock, Sequence2SpatialBlock
from modules.registeration_module import FeatureRegisterationModule

class Model(nn.Module):
    def __init__(self, args, load_weights_from=None):
        super().__init__()
        self.args = args
        model = build_model(args)
        self.feature_backbone = FeatureBackbone(args, model)
        self.registeration_module = FeatureRegisterationModule(args)
        self.bicubic_resize = K.augmentation.Resize((64, 64), resample=2, keepdim=True)
        self.unet_encoder = nn.ModuleList([DownSamplingBlock(i, j) for i, j in args.decoder.downsampling_blocks])
        self.unet_decoder = UnetDecoder(
            encoder_channels=args.decoder.encoder_channels,
            decoder_channels=args.decoder.decoder_channels,
            n_blocks=len(args.decoder.decoder_channels),
            use_batchnorm=True,
            center=False,
            attention_type="scse",
            num_coam_layers=0,
            return_features=False,
        )
        self.feature_fusion_block = FeatureFusionBlock(input_dims=64 + 768, hidden_dims=256, output_dims=64, output_resolution=[224,224])
        self.centernet_head = CenterNetHead(
            in_channel=64,
            feat_channel=64,
            num_classes=1,
            test_cfg=EasyDict({"topk": 100, "local_maximum_kernel": 3, "max_per_img": 100}),
        )
        self.centernet_head.init_weights()
        if load_weights_from is not None:
            self.safely_load_state_dict(torch.load(load_weights_from))

    def safely_load_state_dict(self, checkpoint_state_dict):
        model_state_dict = self.state_dict()
        for k in checkpoint_state_dict:
            if k in model_state_dict:
                if checkpoint_state_dict[k].shape != model_state_dict[k].shape:
                    L.log(
                        "INFO",
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {checkpoint_state_dict[k].shape}",
                    )
                    checkpoint_state_dict[k] = model_state_dict[k]
            else:
                L.log("INFO", f"Dropping parameter {k}")
        self.load_state_dict(checkpoint_state_dict, strict=False)

    def forward(self, batch):
        image1_dino_features = self.feature_backbone(batch["image1"])
        image2_dino_features = self.feature_backbone(batch["image2"])
        image1_last_layer = self.bicubic_resize(image1_dino_features[-1])
        image2_last_layer = self.bicubic_resize(image2_dino_features[-1])
        image1_encoded_features = [[], image1_last_layer]
        image2_encoded_features = [[], image2_last_layer]
        for layer in self.unet_encoder:
            image1_encoded_features.append(layer(image1_encoded_features[-1]))
            image2_encoded_features.append(layer(image2_encoded_features[-1]))
        for i in range(len(self.unet_encoder)+1):
            image1_encoded_features[i + 1], image2_encoded_features[i + 1] = self.registeration_module(
                batch, image1_encoded_features[i + 1], image2_encoded_features[i + 1]
            )
        image1_decoded_features = self.unet_decoder(*image1_encoded_features)
        image2_decoded_features = self.unet_decoder(*image2_encoded_features)
        image1_decoded_features = self.feature_fusion_block(image1_dino_features[0], image1_decoded_features)
        image2_decoded_features = self.feature_fusion_block(image2_dino_features[0], image2_decoded_features)
        return (
            self.centernet_head([image1_decoded_features]),
            self.centernet_head([image2_decoded_features]),
        )

    def get_bboxes_from_logits(self, image1_outputs, image2_outputs, batch):
        image1_predicted_bboxes = self.centernet_head.get_bboxes(
            *image1_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        image2_predicted_bboxes = self.centernet_head.get_bboxes(
            *image2_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        return image1_predicted_bboxes, image2_predicted_bboxes

    @torch.no_grad()
    def predict(self, batch):
        image1_outputs, image2_outputs = self(batch)
        batch_image1_predicted_bboxes, batch_image2_predicted_bboxes = self.get_bboxes_from_logits(image1_outputs, image2_outputs, batch)
        return batch_image1_predicted_bboxes, batch_image2_predicted_bboxes

    def compute_loss(self, batch, image1_outputs, image2_outputs):
        image1_losses = self.centernet_head.loss(
            *image1_outputs,
            batch["target_bbox_1"],
            batch["target_bbox_labels1"],
            img_metas=batch["query_metadata"],
        )
        image2_losses = self.centernet_head.loss(
            *image2_outputs,
            batch["target_bbox_2"],
            batch["target_bbox_labels2"],
            img_metas=batch["query_metadata"],
        )
        overall_loss = 0
        for key in image1_losses:
            overall_loss += image1_losses[key] + image2_losses[key]
        return overall_loss

class FeatureBackbone(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.sequence_to_spatial = nn.ModuleList([Sequence2SpatialBlock(args) for _ in args.vit_feature_layers])
        self._features = []
        self.register_hooks(args.vit_feature_layers)

    def register_hooks(self, hook_layers):
        for index in hook_layers:

            def _hook(module, input, output):
                qkv = rearrange(output, "b n (t c) -> t b n c", t=3)
                self._features.append(qkv[1])

            self.model.blocks[index].attn.qkv.register_forward_hook(_hook)

    def forward(self, x):
        self.model.forward_features(x)  # desired features will get stored in self._features
        output = [self.sequence_to_spatial[i](feature) for i, feature in enumerate(self._features)]
        self._features.clear()  # clear for next forward pass
        return output

def build_model(args, frozen=True):
    model = timm.create_model("vit_base_patch8_224_dino", pretrained=True)
    model = patch_vit_resolution(model, image_hw=[224,224], stride=args.encoder.stride)
    if frozen:
        for _, value in model.named_parameters():
            value.requires_grad = False
    return model


def patch_vit_resolution(model: nn.Module, image_hw, stride: int) -> nn.Module:
    """
    change resolution of model output by changing the stride of the patch extraction.
    :param model: the model to change resolution for.
    :param stride: the new stride parameter.
    :return: the adjusted model
    """
    patch_size = model.patch_embed.patch_size
    if stride == patch_size:  # nothing to do
        return model

    stride = nn_utils._pair(stride)
    assert all([(p // s_) * s_ == p for p, s_ in zip(patch_size, stride)]), f"stride {stride} should divide patch_size {patch_size}"

    # fix the stride
    model.patch_embed.proj.stride = stride
    # fix the positional encoding code
    model._pos_embed = types.MethodType(fix_pos_enc(patch_size, image_hw, stride), model)
    return model

def fix_pos_enc(patch_size: Tuple[int, int], image_hw, stride_hw: Tuple[int, int]):
    """
    Creates a method for position encoding interpolation.
    :param patch_size: patch size of the model.
    :param stride_hw: A tuple containing the new height and width stride respectively.
    :return: the interpolation method
    """

    def interpolate_pos_encoding(self, x) -> torch.Tensor:
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        h, w = image_hw
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        # compute number of tokens taking stride into account
        w0 = 1 + (w - patch_size[1]) // stride_hw[1]
        h0 = 1 + (h - patch_size[1]) // stride_hw[0]
        assert (
            w0 * h0 == npatch
        ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and
                                        stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=False,
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return x + torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    return interpolate_pos_encoding