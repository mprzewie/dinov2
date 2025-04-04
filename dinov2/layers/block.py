# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from .attention import Attention, MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


logger = logging.getLogger("dinov2")

APPLY_TO_ALL = "qkvpr1r2"

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Block)")
    else:
        warnings.warn("xFormers is disabled (Block)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False

    warnings.warn("xFormers is not available (Block)")


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, return_attention: bool = False) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            if not return_attention or isinstance(self.attn, MemEffAttention):
                return self.ls1(self.attn(self.norm1(x)))
            elif isinstance(self.attn, Attention):
                x, attn = self.attn(self.norm1(x), return_attention=True)
                x = self.ls1(x)
                return x, attn

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x_attn = attn_residual_func(x)
            if isinstance(x_attn, Tensor):
                xattn = x_attn
            else:
                xattn, attn = x_attn

            x = x + xattn
            x = x + ffn_residual_func(x)
            if return_attention:
                return x, attn

        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError


class OrthogonalLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, num_reflections=1):
        super().__init__()
        # assert in_features == out_features
        # features = in_features
        self.in_features = in_features
        self.out_features = out_features

        features = max(in_features, out_features)

        # self.num_chunks = num_reflections
        assert num_reflections == 1

        # Householder vector (N-1 parameters)
        self.v = nn.Parameter(torch.randn(features - 1))

        # Rotation vectors for each chunk (num_chunks x (N-1))
        self.r = nn.Parameter(torch.randn(features - 1) * 0.1)
        # self.register_buffer("r", torch.zeros(features - 1))

        # Modulation vector (N parameters)
        self.m = nn.Parameter(torch.ones(out_features))


        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def construct_W(self):
        """Fully vectorized construction of K orthogonal matrices, outputting only the necessary rows."""
        N = max(self.in_features, self.out_features)
        # chunk_size = N // self.num_chunks  # Each chunk contributes this many rows
        device = self.r.device

        # Step 1: Construct Skew-Symmetric Rotation Matrices for Each Chunk
        R = torch.zeros((N, N), device=device)  # Shape: (K, N, N)

        # Correctly assign N-1 parameters per chunk to ensure skew-symmetry
        indices = torch.arange(1, N, device=device)

        # Fix: Now `self.r` is of shape (K, N-1) to allow independent rotations per chunk
        R[0, indices] = self.r  # Rotate e2,...,eN around e1 for each chunk
        R[indices, 0] = -self.r  # Ensure skew-symmetry

        # # Compute full batch of orthogonal rotation matrices
        # Q_rotated = torch.matrix_exp(R)  # Shape: (K, N, N)

        # Rodrigues' Formula
        theta = self.r.norm() + 1e-8
        A = R / theta
        Q_rot2 = (
                torch.eye(R.shape[0], device=R.device)
                + torch.sin(theta) * A
                + (1 - torch.cos(theta)) * (A @ A)
        )
        Q_rotated = Q_rot2
        # assert torch.allclose(Q_rotated, Q_rot2, rtol=1e-4), (Q_rotated - Q_rot2).abs().max()


        # Step 2: Compute Householder Reflection (Fixing e1 -> v1)
        v_full = torch.cat([torch.tensor([1.0], device=device), self.v])  # Extend to full size
        v_full = v_full / v_full.norm()  # Normalize to be a unit vector
        H = torch.eye(N, device=device) - 2 * torch.outer(v_full, v_full)  # Householder matrix

        # Step 3: Apply Householder Reflection After Rotation
        W = H @ Q_rotated

        # Step 6: Cut out the relevant part of constructed W
        W = W[:self.out_features, :self.in_features]

        # Step 7: Apply Modulation
        W = W * self.m.unsqueeze(1)  # Apply modulation

        return W

    def forward(self, x):
        W = self.construct_W()
        return torch.nn.functional.linear(x, W, self.bias)

class QKV(torch.nn.Module):
    def __init__(self, dim: int, bias: bool=True, num_reflections: int=1, apply_to: str = APPLY_TO_ALL):
        super().__init__()
        self.q = OrthogonalLinear(dim, dim, bias=bias, num_reflections=num_reflections)  if "q" in apply_to else nn.Linear(dim, dim, bias=bias)
        self.k = OrthogonalLinear(dim, dim, bias=bias, num_reflections=num_reflections)  if "k" in apply_to else nn.Linear(dim, dim, bias=bias)
        self.v = OrthogonalLinear(dim, dim, bias=bias, num_reflections=num_reflections)  if "v" in apply_to else nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return torch.cat([q,k,v], dim=-1)


class OrtoAttention(MemEffAttention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias: bool = True,  proj_drop=0., attn_drop=0., orto_reflections: int = 0, apply_to: str = APPLY_TO_ALL):
        super().__init__(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        if orto_reflections > 0:
            self.qkv = QKV(dim, bias=qkv_bias, num_reflections=orto_reflections, apply_to=apply_to)
            if "p" in apply_to:
                self.proj = OrthogonalLinear(dim, dim, num_reflections=orto_reflections, bias=proj_bias)


class OrtoMlp(Mlp):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0., orto_reflections: int = 0, apply_to: str="r1r2"):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            bias=bias,
            drop=drop,
        )
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # bias = to_2tuple(bias)

        if orto_reflections > 0:
            if "r1" in apply_to:
                self.fc1 = OrthogonalLinear(in_features, hidden_features, bias=bias, num_reflections=orto_reflections)
            if "r2" in apply_to:
                self.fc2 = OrthogonalLinear(hidden_features, out_features, bias=bias, num_reflections=orto_reflections)


class OrtoBlock(NestedTensorBlock):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            ffn_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            init_values=None,
            drop_path: float = 0.0,
            act_layer: Callable[..., nn.Module] = nn.GELU,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            attn_class: Callable[..., nn.Module] = Attention,
            ffn_layer: Callable[..., nn.Module] = Mlp,
            orto_reflections: int = 0,
            orto_apply_to: str = APPLY_TO_ALL
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attn_class=attn_class,
            ffn_layer=ffn_layer,
            # mlp_layer=mlp_layer
        )
        self.attn = OrtoAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop, orto_reflections=orto_reflections, apply_to=orto_apply_to)
        if "r" in orto_apply_to:
            self.mlp = OrtoMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop, orto_reflections=orto_reflections)
