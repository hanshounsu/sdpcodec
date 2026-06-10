from collections.abc import Sequence
from contextlib import nullcontext
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.amp import autocast

from .module import EncoderBlock, WNConv1d
from .vocos_decoder import TimeTransformerBlock1d, _build_token_norm, _resolve_attn_window_size


class _ChannelLastNorm1d(nn.Module):
    def __init__(self, dim: int, norm_type: str, norm_eps: float):
        super().__init__()
        self.norm = _build_token_norm(int(dim), norm_type=norm_type, norm_eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def _build_pointwise_activation(name: Optional[str]) -> nn.Module:
    resolved = "none" if name is None else str(name).lower()
    if resolved in {"none", "identity", "linear"}:
        return nn.Identity()
    if resolved == "silu":
        return nn.SiLU()
    if resolved == "gelu":
        return nn.GELU()
    if resolved == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    raise ValueError(f"Unsupported VocosFormer codec encoder downsample_activation: {name}")


class _ConvExpandDownsample1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int,
        *,
        norm_type: str,
        norm_eps: float,
        activation: str = "silu",
    ):
        super().__init__()
        stride = int(stride)
        if stride < 1:
            raise ValueError(f"Invalid conv_expand stride={stride}")
        if stride == 1:
            kernel_size = 3
            padding = 1
        else:
            kernel_size = 2 * stride
            padding = stride // 2 + stride % 2
        self.conv = WNConv1d(
            int(input_channels),
            int(output_channels),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = _ChannelLastNorm1d(int(output_channels), norm_type=norm_type, norm_eps=norm_eps)
        self.activation = _build_pointwise_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


class VocosFormerCodecEncoderWrapper(nn.Module):
    """
    Codec encoder adapter for SSL features.

    The upstream SSL model, e.g. HuBERT, still extracts the content features.
    This wrapper replaces the post-SSL Conv/RNN codec adapter with a shallow
    VocosFormer-style time transformer stack.
    """

    def __init__(
        self,
        encoder: nn.Module,
        input_channels: int,
        out_channels: int = 512,
        up_ratios: Optional[Sequence[int]] = None,
        dilations: Sequence[int] = (1, 3, 9),
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 2,
        num_heads: int = 8,
        window_size: Optional[Union[int, Sequence[int]]] = None,
        attn_window_size: Optional[Sequence[int]] = None,
        attention_impl: str = "flash_attn",
        use_rope: bool = True,
        rope_base: float = 10000.0,
        max_position_embeddings: int = 2048,
        qk_norm: bool = True,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-2,
        dropout: float = 0.0,
        ffn_type: str = "swiglu",
        ffn_mult: float = 4.0,
        layerscale_gamma_init: float = 1.0,
        encoder_force_fp32: bool = False,
        activation_type: str = "SnakeBeta",
        leaky_relu_params: Optional[dict] = None,
        snake_lite_taylor_degree: int = 8,
        downsample_mode: str = "conv",
        pre_num_layers: int = 0,
        post_num_layers: Optional[int] = None,
        downsample_dim: Optional[int] = None,
        downsample_activation: str = "silu",
    ):
        super().__init__()
        self.encoder = encoder
        self.input_channels = int(input_channels)
        self.out_channels = int(out_channels)
        self.dim = int(dim)
        self.intermediate_dim = int(intermediate_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.up_ratios = None if up_ratios is None else tuple(int(r) for r in up_ratios)
        self.attn_window_size = _resolve_attn_window_size(
            window_size=window_size,
            attn_window_size=attn_window_size,
        )
        self.downsample_mode = str(downsample_mode).lower()
        if self.downsample_mode not in {"conv", "stack", "conv_expand", "conv_expand_downsample_first"}:
            raise ValueError(f"Unsupported VocosFormer codec encoder downsample_mode: {downsample_mode}")
        self.pre_num_layers = max(0, int(pre_num_layers))
        self.post_num_layers = int(self.num_layers if post_num_layers is None else post_num_layers)
        if self.post_num_layers < 0:
            raise ValueError(f"Invalid VocosFormer codec encoder post_num_layers={post_num_layers}")
        self.downsample_dim = int(downsample_dim) if downsample_dim is not None else max(self.input_channels, self.dim)
        self.downsample_activation = str(downsample_activation)
        self.stack_factor = 1
        if self.up_ratios is not None:
            for ratio in self.up_ratios:
                self.stack_factor *= int(ratio)
        if self.stack_factor < 1:
            raise ValueError(f"Invalid VocosFormer codec encoder stack_factor={self.stack_factor}")
        self.encoder_force_fp32 = bool(encoder_force_fp32)
        self.encoder_is_frozen = not any(param.requires_grad for param in self.encoder.parameters())

        def _make_transformer_block(block_dim: int, block_intermediate_dim: Optional[int] = None) -> TimeTransformerBlock1d:
            return TimeTransformerBlock1d(
                dim=int(block_dim),
                num_heads=self.num_heads,
                window_size=None,
                attn_window_size=self.attn_window_size,
                intermediate_dim=int(self.intermediate_dim if block_intermediate_dim is None else block_intermediate_dim),
                dropout=float(dropout),
                layer_scale_init_value=float(layerscale_gamma_init),
                attention_impl=attention_impl,
                use_rope=use_rope,
                rope_base=rope_base,
                max_position_embeddings=max_position_embeddings,
                qk_norm=qk_norm,
                norm_type=norm_type,
                norm_eps=norm_eps,
                ffn_type=ffn_type,
                ffn_mult=ffn_mult,
            )

        self.pre_backbone = nn.ModuleList()
        self.downsample_proj = nn.Identity()
        if self.downsample_mode in {"conv_expand", "conv_expand_downsample_first"}:
            self.input_proj = nn.Identity()
            if self.downsample_mode == "conv_expand":
                self.pre_backbone = nn.ModuleList(
                    [
                        _make_transformer_block(
                            self.input_channels,
                            block_intermediate_dim=max(self.intermediate_dim, self.input_channels * self.intermediate_dim // max(1, self.dim)),
                        )
                        for _ in range(self.pre_num_layers)
                    ]
                )
            downsample_layers = []
            current_channels = self.input_channels
            if self.up_ratios is not None and len(self.up_ratios) > 0:
                for stride in self.up_ratios:
                    downsample_layers.append(
                        _ConvExpandDownsample1d(
                            input_channels=current_channels,
                            output_channels=self.downsample_dim,
                            stride=stride,
                            norm_type=norm_type,
                            norm_eps=norm_eps,
                            activation=self.downsample_activation,
                        )
                    )
                    current_channels = self.downsample_dim
            self.downsample = nn.Sequential(*downsample_layers)
            projection_kernel = 3 if self.up_ratios is not None and len(self.up_ratios) > 0 else 1
            projection_padding = projection_kernel // 2
            self.downsample_proj = WNConv1d(
                current_channels,
                self.dim,
                kernel_size=projection_kernel,
                padding=projection_padding,
            )
            backbone_layers = self.pre_num_layers + self.post_num_layers if self.downsample_mode == "conv_expand_downsample_first" else self.post_num_layers
        else:
            proj_input_channels = self.input_channels * self.stack_factor if self.downsample_mode == "stack" else self.input_channels
            self.input_proj = nn.Conv1d(proj_input_channels, self.dim, kernel_size=1)
            downsample_layers = []
            if self.downsample_mode == "conv" and self.up_ratios is not None:
                for stride in self.up_ratios:
                    downsample_layers.append(
                        EncoderBlock(
                            dim=self.dim,
                            input_dim=self.dim,
                            stride=stride,
                            dilations=dilations,
                            activation_type=activation_type,
                            leaky_relu_params=leaky_relu_params,
                            snake_lite_taylor_degree=snake_lite_taylor_degree,
                        )
                    )
            self.downsample = nn.Sequential(*downsample_layers)
            backbone_layers = self.num_layers
        self.backbone = nn.ModuleList(
            [
                _make_transformer_block(self.dim)
                for _ in range(backbone_layers)
            ]
        )
        self.final_norm = _build_token_norm(self.dim, norm_type=norm_type, norm_eps=norm_eps)
        self.output_proj = nn.Conv1d(self.dim, self.out_channels, kernel_size=1)
        self.reset_parameters()

        print(
            "VocosFormerCodecEncoderWrapper init: "
            f"input_channels={self.input_channels}, dim={self.dim}, out_channels={self.out_channels}, "
            f"up_ratios={self.up_ratios}, intermediate_dim={self.intermediate_dim}, "
            f"num_layers={self.num_layers}, pre_num_layers={self.pre_num_layers}, "
            f"post_num_layers={len(self.backbone)}, num_heads={self.num_heads}, "
            f"downsample_mode={self.downsample_mode}, stack_factor={self.stack_factor}, "
            f"downsample_dim={self.downsample_dim}, downsample_activation={self.downsample_activation}, "
            f"window_size={self.attn_window_size}, attention_impl={attention_impl}, "
            f"use_rope={use_rope}, qk_norm={qk_norm}, norm_type={norm_type}, norm_eps={norm_eps}, "
            f"ffn_type={ffn_type}, ffn_mult={ffn_mult}"
        )

    def reset_parameters(self) -> None:
        for module in (self.input_proj, self.output_proj):
            if not hasattr(module, "weight"):
                continue
            nn.init.trunc_normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, spk_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        del spk_cond
        use_fp32_island = (
            x.is_cuda
            and self.encoder_is_frozen
            and self.encoder_force_fp32
        )
        autocast_ctx = (
            autocast(device_type=x.device.type, enabled=False)
            if use_fp32_island
            else nullcontext()
        )
        if self.encoder_is_frozen:
            self.encoder.eval()
        with autocast_ctx:
            x = self.encoder(x)
        if isinstance(x, (tuple, list)):
            x = x[0]
        if self.downsample_mode == "stack" and self.stack_factor > 1:
            x = self._stack_adjacent_frames(x, self.stack_factor)
        x = self.input_proj(x)
        for block in self.pre_backbone:
            x = block(x)
        x = self.downsample(x)
        x = self.downsample_proj(x)
        for block in self.backbone:
            x = block(x)
        x = self.final_norm(x.transpose(1, 2)).transpose(1, 2)
        return self.output_proj(x)

    @staticmethod
    def _stack_adjacent_frames(x: torch.Tensor, factor: int) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected SSL features as (B, C, T), got shape={tuple(x.shape)}")
        if factor <= 1:
            return x
        batch, channels, frames = x.shape
        remainder = frames % factor
        if remainder:
            pad_frames = factor - remainder
            x = torch.cat([x, x[:, :, -1:].expand(batch, channels, pad_frames)], dim=-1)
            frames = x.shape[-1]
        x = x.reshape(batch, channels, frames // factor, factor)
        x = x.permute(0, 1, 3, 2).reshape(batch, channels * factor, frames // factor)
        return x
