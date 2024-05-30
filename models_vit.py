# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from torchvision import transforms
import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        # print((x == 0).all())
        B = x.shape[0]
        x = self.patch_embed(x)
        # print((x == 0).all())

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # # print(cls_tokens)
        # # print(cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        # # print((x == 0).all())
        x = x + self.pos_embed
        # print(self.pos_embed)
        x = self.pos_drop(x)
        
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # print(f"block{i}", x.mean())

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        # print("outcome", outcome)
        return outcome


class VisionTransformerDVS(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, in_channels_dvs=18, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), **kwargs):
        super(VisionTransformerDVS, self).__init__(**kwargs)

        self.align = nn.Conv2d(in_channels=in_channels_dvs, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.global_pool = global_pool
        self.mean = mean
        self.std = std
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        # print((x == 0).all())
        B = x.shape[0]
        x = self.align(x)
        # x = transforms.functional.normalize(x, self.mean, self.std)
        x = self.patch_embed(x)
        # print((x == 0).all())

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # # print(cls_tokens)
        # # print(cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        # # print((x == 0).all())
        x = x + self.pos_embed
        # print(self.pos_embed)
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            # print(i, x.shape)
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
         **kwargs) # remenber the modify
    return model

def vit_small_patch16_dvs(**kwargs):
    model = VisionTransformerDVS(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

if __name__ == '__main__':
    model = vit_small_patch16(act_layer=nn.ReLU)
    d = torch.load("pretrained/deit-small-pretrained.pth")["model"]
    model.load_state_dict(d)
    # import torch
    # # check you have the right version of timm
    # import timm
    #
    # assert timm.__version__ == "0.3.2"
    #
    # # now load it with torchhub
    # model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)
    # print(model.blocks[0].attn.num_heads)
