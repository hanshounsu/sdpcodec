"""
F0 Codec using CodecEncoder/CodecDecoder structure.
This is an alternative to the Jukebox-style F0Encoder/F0Decoder.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from torch.amp import autocast
from torch.nn.utils.weight_norm import _weight_norm
from ..codec_encoder import CodecEncoder
from ..module import WNConv1d, DecoderBlock, EncoderBlock, CrossAttentionBlock, build_res_temporal
from ..vocos_decoder import ConvDownsample, ConvUpsample, Transformer, _build_token_norm
from ..alias_free_torch import Activation1d
from .. import activations
from termcolor import colored


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        if m.bias is not None:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)


def _tensor_debug_stats(x: Optional[torch.Tensor]):
    if x is None:
        return None
    with torch.no_grad():
        x_detached = x.detach().float()
        finite_mask = torch.isfinite(x_detached)
        stats = {
            "shape": tuple(x_detached.shape),
            "non_finite": int((~finite_mask).sum().item()),
        }
        if finite_mask.any():
            finite_vals = x_detached[finite_mask]
            stats.update(
                {
                    "min": float(finite_vals.min().item()),
                    "max": float(finite_vals.max().item()),
                    "mean": float(finite_vals.mean().item()),
                    "absmax": float(finite_vals.abs().max().item()),
                }
            )
        return stats


def fcpe_mixture_param_count(components: int = 2) -> int:
    """weight, mean, left sigma, right sigma for each mixture component."""
    return 4 * int(components)


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x * (1.0 / math.sqrt(2.0))))


def _two_piece_normal_cdf(
    x: torch.Tensor,
    mean: torch.Tensor,
    sigma_left: torch.Tensor,
    sigma_right: torch.Tensor,
) -> torch.Tensor:
    sigma_sum = (sigma_left + sigma_right).clamp_min(1.0e-6)
    left_mass = sigma_left / sigma_sum
    left = 2.0 * left_mass * _normal_cdf((x - mean) / sigma_left.clamp_min(1.0e-6))
    right = left_mass + 2.0 * (1.0 - left_mass) * (
        _normal_cdf((x - mean) / sigma_right.clamp_min(1.0e-6)) - 0.5
    )
    return torch.where(x < mean, left, right)


def fcpe_mixture_params_to_distribution(
    raw_params: torch.Tensor,
    num_bins: int = 360,
    components: int = 2,
    min_sigma: float = 0.75,
    max_sigma: float = 80.0,
    eps: float = 1.0e-8,
    sort_means: bool = True,
) -> torch.Tensor:
    """
    Convert compact asymmetric Gaussian-mixture parameters to discretized FCPE bin mass.

    Args:
        raw_params: (B, 4*K, T), split as weight logits, mean raw, left sigma raw,
            right sigma raw.
    Returns:
        Probability mass over FCPE bins, shape (B, num_bins, T).
    """
    k = int(components)
    expected = fcpe_mixture_param_count(k)
    if raw_params.dim() != 3 or raw_params.shape[1] != expected:
        raise ValueError(
            f"Expected raw_params shape (B, {expected}, T) for K={k}, got {tuple(raw_params.shape)}"
        )

    weight_logits, mean_raw, sigma_left_raw, sigma_right_raw = raw_params.float().chunk(4, dim=1)
    mean = torch.sigmoid(mean_raw) * float(max(int(num_bins) - 1, 1))
    sigma_left = float(min_sigma) + F.softplus(sigma_left_raw)
    sigma_right = float(min_sigma) + F.softplus(sigma_right_raw)
    sigma_left = sigma_left.clamp(max=float(max_sigma))
    sigma_right = sigma_right.clamp(max=float(max_sigma))

    if sort_means and k > 1:
        order = mean.argsort(dim=1)
        mean = torch.gather(mean, 1, order)
        weight_logits = torch.gather(weight_logits, 1, order)
        sigma_left = torch.gather(sigma_left, 1, order)
        sigma_right = torch.gather(sigma_right, 1, order)

    weights = torch.softmax(weight_logits, dim=1)
    edges = torch.arange(
        int(num_bins) + 1,
        device=raw_params.device,
        dtype=raw_params.float().dtype,
    ).sub_(0.5)
    edges = edges.view(1, 1, -1, 1)
    mean = mean.unsqueeze(2)
    sigma_left = sigma_left.unsqueeze(2)
    sigma_right = sigma_right.unsqueeze(2)

    cdf = _two_piece_normal_cdf(edges, mean, sigma_left, sigma_right)
    component_mass = (cdf[:, :, 1:, :] - cdf[:, :, :-1, :]).clamp_min(float(eps))
    prob = (weights.unsqueeze(2) * component_mass).sum(dim=1)
    prob = prob / prob.sum(dim=1, keepdim=True).clamp_min(float(eps))
    return prob.clamp_min(float(eps))


def normalize_fcpe_loss_mode(fcpe_loss_mode=None, legacy_use_fcpe_loss=None) -> str:
    if fcpe_loss_mode is None:
        return 'dense' if bool(legacy_use_fcpe_loss) else 'none'
    mode = str(fcpe_loss_mode).strip().lower()
    if mode in {'', 'none', 'off', 'false', '0', 'disabled'}:
        return 'none'
    if mode in {'true', '1', 'on', 'enabled', 'bce', 'wbce'}:
        return 'dense'
    if mode in {
        'dense',
        'discretized_mixture',
        'mixture',
        'asym_discretized_mixture',
        'asymmetric_discretized_mixture',
    }:
        return mode
    raise ValueError(f"Unsupported fcpe_loss_mode: {fcpe_loss_mode}")


def fcpe_loss_mode_enabled(fcpe_loss_mode: str) -> bool:
    return normalize_fcpe_loss_mode(fcpe_loss_mode) != 'none'


def build_channel_schedule(ngf, num_levels, max_channels=None):
    channels = []
    current = int(ngf)
    max_channels = None if max_channels is None else int(max_channels)
    for _ in range(int(num_levels)):
        current = current * 2
        if max_channels is not None:
            current = min(current, max_channels)
        channels.append(int(current))
    return channels


class ProjectedSpeakerConcatFuser(nn.Module):
    """Exact 1x1 speaker concat path without expanding global speaker features over time."""

    def __init__(self, input_dim: int, condition_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.condition_proj = nn.Linear(condition_dim, self.input_dim)
        self.merge_conv = WNConv1d(self.input_dim * 2, self.input_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        projected = self.condition_proj(condition)
        weight = _weight_norm(self.merge_conv.weight_v, self.merge_conv.weight_g, 0)
        bias = self.merge_conv.bias
        x_out = F.conv1d(x, weight[:, :self.input_dim, :], bias=bias)
        cond_out = F.linear(projected, weight[:, self.input_dim:, 0]).unsqueeze(-1)
        return x_out + cond_out


class F0CodecEncoder(nn.Module):
    """
    F0 Encoder using CodecEncoder structure.
    Converts F0 input (1 or 2 channels) to latent representation.
    """
    def __init__(self,
                 input_channels=1,  # 1 for f0 only, 2 for f0+vuv
                 ngf=16,
                 max_channels=None,
                 use_rnn=True,
                 rnn_bidirectional=False,
                 rnn_num_layers=2,
                 rnn_type='lstm',
                 rnn_mamba=None,
                 up_ratios=(3, 4, 5, 8),
                 dilations=(1, 3, 9),
                 out_channels=128,
                 activation_type='SnakeBeta',
                 leaky_relu_params=None,
                 ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios
        self.input_channels = input_channels
        self.max_channels = None if max_channels is None else int(max_channels)
        self.stage_channels = build_channel_schedule(ngf, len(up_ratios), max_channels=self.max_channels)
        self.temporal_layer_index = None
        
        # Create first convolution (adapt to input_channels)
        prev_dim = int(ngf)
        self.block = [WNConv1d(input_channels, prev_dim, kernel_size=7, padding=3)]
        
        # Create EncoderBlocks with progressive channel growth capped by max_channels.
        for stage_dim, stride in zip(self.stage_channels, up_ratios):
            self.block += [EncoderBlock(stage_dim, stride=stride, dilations=dilations,
                                       speaker_condition=False,  # F0 doesn't use speaker condition
                                       activation_type=activation_type, 
                                       leaky_relu_params=leaky_relu_params,
                                       input_dim=prev_dim)]
            prev_dim = stage_dim
        d_model = prev_dim
        
        # RNN
        if use_rnn:
            temporal_layer = build_res_temporal(
                d_model,
                rnn_type,
                num_layers=rnn_num_layers,
                bidirectional=rnn_bidirectional,
                mamba=rnn_mamba,
            )
            self.block.append(temporal_layer)
            self.temporal_layer_index = len(self.block) - 1
        
        # Create last convolution
        if activation_type == 'LeakyReLU':
            activation = nn.LeakyReLU(negative_slope=leaky_relu_params['negative_slope'])
        else:
            activation = Activation1d(activation=activations.SnakeBeta(d_model, alpha_logscale=True))
        
        self.block += [
            activation,
            WNConv1d(d_model, out_channels, kernel_size=3, padding=1),
        ]
        
        # Wrap block into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model
        
        # Build width_list for compatibility with F0Decoder
        self.width_list = self._build_width_list()
        
        self.reset_parameters()
        
        print(f"F0CodecEncoder: input_channels={input_channels}, out_channels={out_channels}, "
              f"hop_length={self.hop_length}, ngf={ngf}, up_ratios={up_ratios}, max_channels={self.max_channels}")
        print(f"F0CodecEncoder stage_channels: {self.stage_channels}")
        if self.temporal_layer_index is not None:
            print(
                f"F0CodecEncoder temporal: type={rnn_type}, "
                f"num_layers={rnn_num_layers}, bidirectional={rnn_bidirectional}"
            )
        print(f"Number of parameters in F0CodecEncoder: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")
    
    def _build_width_list(self):
        """
        Build width_list for compatibility with F0Decoder that expects it.
        Should have num_levels items (one per EncoderBlock), NOT num_levels + 1.
        """
        return [list(self.stage_channels)]
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T) where C is 1 or 2 (f0 or f0+vuv)
        Returns:
            (B, out_channels, T')
        """
        # Match F0Encoder behavior: if input_channels==1, use only first channel
        # This allows receiving 2-channel input but only using f0 channel
        if self.input_channels == 1:
            x = x[:, :1, :]  # use only f0 for encoding
        
        return self.block(x)
    
    def reset_parameters(self):
        self.apply(init_weights)


class HoyeolStyleF0CodecEncoder(nn.Module):
    """
    Lightweight F0 encoder using the same Hoyeol block names:
    input_proj -> ConvDownsample(s) -> transformer_backbone_level2
    -> transformer_backbone_level1 -> output_proj.
    """

    def __init__(
        self,
        input_channels=1,
        out_channels=256,
        dim=256,
        intermediate_dim=768,
        downsample_factor=4,
        level2_num_layers=2,
        level1_num_layers=2,
        num_heads=4,
        attn_window_size=(64, 64),
        attention_impl="flash_attn",
        use_rope=True,
        rope_base=10000.0,
        max_position_embeddings=2048,
        qk_norm=True,
        norm_type="rmsnorm",
        norm_eps=1e-2,
        dropout=0.0,
        ffn_type="swiglu",
        ffn_mult=4.0,
        layerscale_gamma_init=1.0,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.out_channels = int(out_channels)
        self.dim = int(dim)
        self.intermediate_dim = int(intermediate_dim)
        self.downsample_factor = int(downsample_factor)
        self.level2_num_layers = int(level2_num_layers)
        self.level1_num_layers = int(level1_num_layers)
        self.num_layers = self.level2_num_layers + self.level1_num_layers
        if self.downsample_factor < 1 or self.downsample_factor & (self.downsample_factor - 1):
            raise ValueError(
                "HoyeolStyleF0CodecEncoder downsample_factor must be a power of two; "
                f"got {self.downsample_factor}."
            )
        self.num_downsamples = int(math.log2(self.downsample_factor))
        if self.num_layers <= 0:
            raise ValueError("HoyeolStyleF0CodecEncoder requires at least one transformer layer.")

        self.input_proj = WNConv1d(self.input_channels, self.dim, kernel_size=7, padding=3)
        self.downsamplers = nn.ModuleList(
            [
                ConvDownsample(
                    self.dim,
                    self.dim,
                    norm_type=norm_type,
                    norm_eps=norm_eps,
                )
                for _ in range(self.num_downsamples)
            ]
        )

        def _make_transformer(n_layers: int) -> Transformer:
            return Transformer(
                n_layers=n_layers,
                dim=self.dim,
                num_heads=int(num_heads),
                window_size=None,
                attn_window_size=attn_window_size,
                intermediate_dim=self.intermediate_dim,
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

        self.transformer_backbone_level2 = _make_transformer(self.level2_num_layers)
        self.transformer_backbone_level1 = _make_transformer(self.level1_num_layers)
        self.final_norm = _build_token_norm(self.dim, norm_type=norm_type, norm_eps=norm_eps)
        self.output_proj = WNConv1d(self.dim, self.out_channels, kernel_size=3, padding=1)
        self.width_list = [[self.dim] * self.num_layers]
        self.enc_dim = self.dim

        self.reset_parameters()
        print(
            "HoyeolStyleF0CodecEncoder: "
            f"input_channels={self.input_channels}, dim={self.dim}, out_channels={self.out_channels}, "
            f"downsample_factor={self.downsample_factor}, num_downsamples={self.num_downsamples}, "
            f"level2_num_layers={self.level2_num_layers}, level1_num_layers={self.level1_num_layers}"
        )
        print(
            "HoyeolStyleF0CodecEncoder stack: "
            "input_proj -> "
            f"ConvDownsample x{self.num_downsamples} -> "
            "transformer_backbone_level2 -> transformer_backbone_level1 -> output_proj"
        )

    def _iter_transformer_blocks(self):
        yield from self.transformer_backbone_level2.layers
        yield from self.transformer_backbone_level1.layers

    def forward(self, x):
        if self.input_channels == 1:
            x = x[:, :1, :]
        x = self.input_proj(x)
        for downsample in self.downsamplers:
            x = downsample(x)
        for block in self._iter_transformer_blocks():
            x = block(x)
        x = self.final_norm(x.transpose(1, 2)).transpose(1, 2)
        return self.output_proj(x)

    def reset_parameters(self):
        def _init(module):
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_init)


class F0CodecDecoder(nn.Module):
    """
    F0 Decoder using CodecDecoder structure (without quantizer).
    Converts latent representation back to F0 output (1 or 2 channels).
    """
    def __init__(self,
                 in_channels=128,
                 upsample_initial_channel=None,  # Auto-calculated if None
                 output_channels=1,  # 1 for f0 only, 2 for f0+vuv
                 ngf=16,
                 max_channels=None,
                 use_rnn=True,
                 rnn_bidirectional=False,
                 rnn_num_layers=2,
                 rnn_type='lstm',
                 rnn_mamba=None,
                 up_ratios=(8, 5, 4, 3),
                 dilations=(1, 3, 9),
                 activation_type='SnakeBeta',
                 leaky_relu_params=None,
                 fcpe_out_dims=360,  # FCPE latent dimension
                 fcpe_loss_mode=None,
                 fcpe_mixture_components=2,
                 fcpe_mixture_min_sigma=0.75,
                 fcpe_mixture_max_sigma=80.0,
                 use_fcpe_loss=None,  # Deprecated compatibility shim; prefer fcpe_loss_mode.
                 speaker_condition=False,  # Use speaker embedding as condition
                 condition_dim=0,  # Speaker embedding dimension
                 use_spk_concat=True,
                 use_spk_film=False,
                 use_mhca=False,
                 mhca_num_heads=2,
                 mhca_dropout=0.1,
                 mhca_key_dim=128,
                 mhca_use_sdpa: Optional[bool] = None,
                 use_split_condition_optimization: bool = True,
                 ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.fcpe_out_dims = fcpe_out_dims
        self.fcpe_loss_mode = normalize_fcpe_loss_mode(fcpe_loss_mode, use_fcpe_loss)
        self.fcpe_loss_enabled = fcpe_loss_mode_enabled(self.fcpe_loss_mode)
        self.use_fcpe_loss = self.fcpe_loss_enabled
        self.fcpe_mixture_components = int(fcpe_mixture_components)
        self.fcpe_mixture_min_sigma = float(fcpe_mixture_min_sigma)
        self.fcpe_mixture_max_sigma = float(fcpe_mixture_max_sigma)
        self.fcpe_uses_mixture = self.fcpe_loss_mode in {
            'discretized_mixture',
            'mixture',
            'asym_discretized_mixture',
            'asymmetric_discretized_mixture',
        }
        self.use_spk_concat = bool(use_spk_concat)
        self.use_spk_film = bool(use_spk_film)
        self.use_split_condition_optimization = bool(use_split_condition_optimization)
        self.speaker_condition = bool(speaker_condition) and (self.use_spk_concat or self.use_spk_film)
        self.condition_dim = condition_dim
        self.use_mhca = use_mhca
        self.mhca_key_dim = mhca_key_dim
        self.max_channels = None if max_channels is None else int(max_channels)
        self.stage_channels = build_channel_schedule(ngf, len(up_ratios), max_channels=self.max_channels)
        self.temporal_layer_index = None
        
        # Auto-calculate upsample_initial_channel to match encoder's final channel
        if upsample_initial_channel is None:
            upsample_initial_channel = self.stage_channels[-1]
        self.upsample_initial_channel = upsample_initial_channel
        self.decoder_output_channels = list(reversed([int(ngf)] + list(self.stage_channels[:-1])))
        self.decoder_input_channels = [self.upsample_initial_channel] + self.decoder_output_channels[:-1]

        channels = self.upsample_initial_channel
        # First conv: no speaker conditioning here (will be added before each DecoderBlock)
        layers = [WNConv1d(in_channels, channels, kernel_size=7, padding=3)]
        
        self.spk_concat_fusers = nn.ModuleList() if self.speaker_condition and self.use_spk_concat else None
        self.spk_proj_layers = None
        self.spk_merge_layers = None
        if self.spk_concat_fusers is not None:
            if self.use_split_condition_optimization:
                for block_input_dim in self.decoder_input_channels:
                    self.spk_concat_fusers.append(
                        ProjectedSpeakerConcatFuser(
                            input_dim=block_input_dim,
                            condition_dim=condition_dim,
                        )
                    )
            else:
                self.spk_proj_layers = nn.ModuleList()
                self.spk_merge_layers = nn.ModuleList()
                for block_input_dim in self.decoder_input_channels:
                    self.spk_proj_layers.append(
                        nn.Linear(condition_dim, block_input_dim)
                    )
                    self.spk_merge_layers.append(
                        WNConv1d(block_input_dim * 2, block_input_dim, kernel_size=1)
                    )
                    self.spk_concat_fusers.append(nn.Identity())

        self.spk_film_layers = nn.ModuleList() if self.speaker_condition and self.use_spk_film else None
        if self.spk_film_layers is not None:
            for block_input_dim in self.decoder_input_channels:
                self.spk_film_layers.append(
                    nn.Linear(condition_dim, block_input_dim * 2)
                )
        
        # RNN
        if use_rnn:
            temporal_layer = build_res_temporal(
                channels,
                rnn_type,
                num_layers=rnn_num_layers,
                bidirectional=rnn_bidirectional,
                mamba=rnn_mamba,
            )
            layers.append(temporal_layer)
            self.temporal_layer_index = len(layers) - 1
        
        # Decoder blocks
        for input_dim, output_dim, stride in zip(self.decoder_input_channels, self.decoder_output_channels, up_ratios):
            layers += [DecoderBlock(input_dim, output_dim, stride, dilations, 
                                   speaker_condition=False,  # F0 doesn't use speaker condition
                                   f0_condition=False,  # F0 decoder doesn't condition on F0
                                   activation_type=activation_type,
                                   leaky_relu_params=leaky_relu_params,
                                   )]
        
        # Final activation and output
        # Skip final scalar-F0 layers when the decoder predicts an FCPE distribution.
        if not self.fcpe_loss_enabled:
            if activation_type == 'LeakyReLU':
                activation = nn.LeakyReLU(negative_slope=leaky_relu_params['negative_slope'])
            else:
                activation = Activation1d(activation=activations.SnakeBeta(output_dim, alpha_logscale=True))
            
            layers += [
                activation,
                WNConv1d(output_dim, output_channels, kernel_size=7, padding=3),
            ]
        
        # Don't use Sequential - we need to access intermediate outputs
        self.layers = nn.ModuleList(layers)
        
        # FCPE-style latent prediction head (optional)
        # Keep logits in fp32 for a more stable BCEWithLogits path.
        if self.fcpe_loss_enabled:
            fcpe_head_dim = (
                fcpe_mixture_param_count(self.fcpe_mixture_components)
                if self.fcpe_uses_mixture
                else self.fcpe_out_dims
            )
            self.fcpe_head = WNConv1d(output_dim, fcpe_head_dim, kernel_size=1)
            self.last_fcpe_logits = None
            self.last_fcpe_mixture_params = None
            self.last_forward_debug = {}
            print(
                "F0CodecDecoder: FCPE loss enabled "
                f"(mode={self.fcpe_loss_mode}, latent_bins={self.fcpe_out_dims}, head_dim={fcpe_head_dim})"
            )
        else:
            self.last_fcpe_mixture_params = None
            self.last_forward_debug = {}
        
        # Cross-Attention Blocks (optional)
        if self.use_mhca:
            self.mhca_list = nn.ModuleList()
            for input_dim in self.decoder_input_channels:
                self.mhca_list.append(
                    CrossAttentionBlock(
                        query_dim=input_dim,
                        key_dim=mhca_key_dim,
                        num_heads=mhca_num_heads,
                        ffn_hidden_dim=None,
                        dropout=mhca_dropout,
                        use_sdpa=mhca_use_sdpa,
                    )
                )
            print(colored(f"F0CodecDecoder: Using Cross-Attention Blocks (key_dim={mhca_key_dim}, heads={mhca_num_heads})", "cyan"))
        else:
            self.mhca_list = None

        # Store indices of DecoderBlocks for later use
        self.decoder_block_indices = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DecoderBlock):
                self.decoder_block_indices.append(i)
        
        # Build width_list for compatibility with CodecDecoder's f0_width_list
        # This represents the channel dimensions at each decoder level
        self.width_list = self._build_width_list()
        
        self.reset_parameters()
        
        print(f"F0CodecDecoder: in_channels={in_channels}, output_channels={output_channels}, "
              f"hop_length={self.hop_length}, ngf={ngf}, upsample_initial_channel={self.upsample_initial_channel}, up_ratios={up_ratios}, max_channels={self.max_channels}")
        print(f"F0CodecDecoder decoder_input_channels: {self.decoder_input_channels}")
        print(f"F0CodecDecoder decoder_output_channels: {self.decoder_output_channels}")
        print(f"F0CodecDecoder width_list: {self.width_list}")
        if self.speaker_condition:
            print(f"F0CodecDecoder: Speaker conditioning enabled (condition_dim={condition_dim})")
            if self.use_spk_concat:
                print(f"  → Speaker concat enabled before each DecoderBlock input ({len(up_ratios)} blocks)")
            if self.use_spk_film:
                print(f"  → Speaker FiLM enabled before each DecoderBlock input ({len(up_ratios)} blocks)")
        if self.temporal_layer_index is not None:
            print(
                f"F0CodecDecoder temporal: type={rnn_type}, "
                f"num_layers={rnn_num_layers}, bidirectional={rnn_bidirectional}"
            )
        print(f"Number of parameters in F0CodecDecoder: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")
    
    def _build_width_list(self):
        """
        Build width_list compatible with CodecDecoder conditioning.
        reversed(width_list[0]) should match:
        [after_rnn] + [decoder block outputs except the last block]
        """
        valid_stage_dims = [self.upsample_initial_channel] + self.decoder_output_channels[:-1]
        return [list(reversed(valid_stage_dims))]
    
    def forward(self, x, spk_emb=None):
        """
        Args:
            x: (B, in_channels, T')
            spk_emb: (B, condition_dim) speaker embedding (optional, required if speaker_condition=True)
                     If use_mhca=True, this can be a tuple (spk_emb, mhca_key) where
                     mhca_key is a time-varying sequence for cross-attention (e.g., x_quantized).
        Returns:
            if self.fcpe_loss_enabled:
                tuple: (outs, fcpe_latent)
                - outs: list of outputs for F0 conditioning
                - fcpe_latent: (B, fcpe_out_dims, T) probabilities for FCPE decoding/monitoring
            else:
                outs: list of outputs for F0 conditioning
        """
        # Split speaker embedding and MHCA key if tuple provided
        if self.use_mhca and isinstance(spk_emb, tuple):
            spk_emb_global, mhca_key = spk_emb
        else:
            spk_emb_global = spk_emb
            mhca_key = None
        
        # Validate speaker embedding if conditioning is enabled
        if self.speaker_condition:
            assert spk_emb_global is not None, "Speaker embedding is required when speaker_condition=True"
        
        # Match original F0Decoder format from codec.py line 185-197
        outs = []
        debug_info = {
            "input": _tensor_debug_stats(x),
            "speaker": _tensor_debug_stats(spk_emb_global) if isinstance(spk_emb_global, torch.Tensor) else None,
            "mhca_key": _tensor_debug_stats(mhca_key) if isinstance(mhca_key, torch.Tensor) else None,
        }
        
        decoder_block_count = 0
        mhca_idx = 0
        
        # Process through all layers
        for i, layer in enumerate(self.layers):
            if self.temporal_layer_index is not None and i == self.temporal_layer_index:
                # After the temporal block, append to outs (this becomes outs[0]).
                x = layer(x)
                debug_info["after_temporal"] = _tensor_debug_stats(x)
                outs.append(x)
            elif isinstance(layer, DecoderBlock):
                # Optional MHCA before each DecoderBlock
                if self.use_mhca and mhca_key is not None:
                    x = self.mhca_list[mhca_idx](x, mhca_key)
                    debug_info[f"after_mhca_{mhca_idx}"] = _tensor_debug_stats(x)
                    mhca_idx += 1

                if self.speaker_condition and self.use_spk_film:
                    gamma_beta = self.spk_film_layers[decoder_block_count](spk_emb_global)
                    gamma, beta = gamma_beta.chunk(2, dim=1)
                    gamma = gamma.unsqueeze(-1)
                    beta = beta.unsqueeze(-1)
                    x = x * (1.0 + gamma) + beta

                # Concatenate speaker embedding BEFORE each DecoderBlock
                if self.speaker_condition and self.use_spk_concat:
                    if self.use_split_condition_optimization:
                        x = self.spk_concat_fusers[decoder_block_count](x, spk_emb_global)
                    else:
                        # Project speaker embedding: (B, condition_dim) -> (B, block_input_dim)
                        spk_emb_proj = self.spk_proj_layers[decoder_block_count](spk_emb_global)  # (B, block_input_dim)
                        # Broadcast to match temporal dimension: (B, block_input_dim) -> (B, block_input_dim, T)
                        spk_emb_broadcast = spk_emb_proj.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                        # Concatenate: (B, block_input_dim, T) + (B, block_input_dim, T) -> (B, 2*block_input_dim, T)
                        x_with_spk = torch.cat([x, spk_emb_broadcast], dim=1)
                        # Merge back to block_input_dim using 1x1 conv
                        x = self.spk_merge_layers[decoder_block_count](x_with_spk)  # (B, block_input_dim, T)
                
                decoder_block_count += 1
                # DecoderBlock.block = [activation, WNConvTranspose1d (index 1), ResidualUnit, ...]
                # Collect output after the stride-changing conv. Skip the final block output
                # to match the original conditioning convention used by CodecDecoder.
                for j, sub_layer in enumerate(layer.block):
                    x = sub_layer(x)
                    if j == 1 and decoder_block_count < len(self.up_ratios):
                        debug_info[f"after_decoder_block_{decoder_block_count}_stride"] = _tensor_debug_stats(x)
                        outs.append(x)
            else:
                # Regular layers (WNConv1d, Activation, etc.)
                x = layer(x)
        
        # Skip one index (to match original structure at index 4)
        outs.append(None)  # outs[4]: placeholder (not used)
        
        # Handle final output based on fcpe_loss_mode.
        if self.fcpe_loss_enabled:
            # x is now the last DecoderBlock output (B, 16, T)
            # Run the FCPE head in fp32 and expose either dense logits or mixture params.
            debug_info["pre_fcpe"] = _tensor_debug_stats(x)
            with autocast(device_type=x.device.type, enabled=False):
                fcpe_head_out = self.fcpe_head(x.float())
                if self.fcpe_uses_mixture:
                    fcpe_latent = fcpe_mixture_params_to_distribution(
                        fcpe_head_out,
                        num_bins=self.fcpe_out_dims,
                        components=self.fcpe_mixture_components,
                        min_sigma=self.fcpe_mixture_min_sigma,
                        max_sigma=self.fcpe_mixture_max_sigma,
                    )
                    fcpe_logits = fcpe_latent.clamp_min(1.0e-8).log()
                    self.last_fcpe_mixture_params = fcpe_head_out
                else:
                    fcpe_logits = fcpe_head_out
                    fcpe_latent = torch.sigmoid(fcpe_logits)
                    self.last_fcpe_mixture_params = None
            self.last_fcpe_logits = fcpe_logits
            debug_info["fcpe_logits"] = _tensor_debug_stats(fcpe_logits)
            debug_info["fcpe_mixture_params"] = (
                _tensor_debug_stats(self.last_fcpe_mixture_params)
                if self.last_fcpe_mixture_params is not None
                else None
            )
            self.last_forward_debug = debug_info
            # For audio conditioning, we still need some F0 output
            # Use a dummy placeholder (won't be used for loss, only for conditioning)
            outs.append(None)  # outs[5]: placeholder (FCPE mode doesn't need final F0)
            return outs, fcpe_latent
        else:
            self.last_fcpe_logits = None
            self.last_fcpe_mixture_params = None
            debug_info["final_output"] = _tensor_debug_stats(x)
            self.last_forward_debug = debug_info
            # Normal mode: x is the final F0 reconstruction after activation + conv
            outs.append(x)  # outs[5]: final F0 reconstruction
            return outs
    
    def reset_parameters(self):
        self.apply(init_weights)


class HoyeolStyleF0CodecDecoder(nn.Module):
    """
    F0 decoder with the same block order as the Hoyeol waveform decoder:
    input projection -> transformer_backbone_level2 -> transformer_backbone_level1
    -> one or more ConvUpsample stages -> reconstruction head.
    """

    def __init__(
        self,
        in_channels=512,
        output_channels=1,
        dim=512,
        intermediate_dim=1536,
        level2_num_layers=2,
        level1_num_layers=2,
        upsample_factor=4,
        fcpe_out_dims=360,
        fcpe_loss_mode=None,
        fcpe_mixture_components=2,
        fcpe_mixture_min_sigma=0.75,
        fcpe_mixture_max_sigma=80.0,
        use_fcpe_loss=None,  # Deprecated compatibility shim; prefer fcpe_loss_mode.
        speaker_condition=False,
        condition_dim=0,
        use_spk_concat=True,
        use_spk_film=False,
        use_mhca=False,
        mhca_num_heads=2,
        mhca_dropout=0.1,
        mhca_key_dim=128,
        mhca_use_sdpa: Optional[bool] = None,
        num_heads=8,
        attn_window_size=(64, 64),
        attention_impl="flash_attn",
        use_rope=True,
        rope_base=10000.0,
        max_position_embeddings=2048,
        qk_norm=True,
        norm_type="rmsnorm",
        norm_eps=1e-2,
        dropout=0.0,
        ffn_type="swiglu",
        ffn_mult=4.0,
        layerscale_gamma_init=1.0,
        stage_output_repeat=2,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.output_channels = int(output_channels)
        self.dim = int(dim)
        self.intermediate_dim = int(intermediate_dim)
        self.level2_num_layers = int(level2_num_layers)
        self.level1_num_layers = int(level1_num_layers)
        self.num_layers = self.level2_num_layers + self.level1_num_layers
        self.upsample_factor = int(upsample_factor)
        self.fcpe_out_dims = int(fcpe_out_dims)
        self.fcpe_loss_mode = normalize_fcpe_loss_mode(fcpe_loss_mode, use_fcpe_loss)
        self.fcpe_loss_enabled = fcpe_loss_mode_enabled(self.fcpe_loss_mode)
        self.use_fcpe_loss = self.fcpe_loss_enabled
        self.fcpe_mixture_components = int(fcpe_mixture_components)
        self.fcpe_mixture_min_sigma = float(fcpe_mixture_min_sigma)
        self.fcpe_mixture_max_sigma = float(fcpe_mixture_max_sigma)
        self.fcpe_uses_mixture = self.fcpe_loss_mode in {
            'discretized_mixture',
            'mixture',
            'asym_discretized_mixture',
            'asymmetric_discretized_mixture',
        }
        self.speaker_condition = bool(speaker_condition) and (bool(use_spk_concat) or bool(use_spk_film))
        self.use_spk_concat = bool(use_spk_concat)
        self.use_spk_film = bool(use_spk_film)
        self.condition_dim = int(condition_dim)
        self.use_mhca = bool(use_mhca)
        self.stage_output_repeat = max(1, int(stage_output_repeat))

        if self.num_layers <= 0:
            raise ValueError("HoyeolStyleF0CodecDecoder requires at least one transformer layer.")
        if self.upsample_factor < 1 or self.upsample_factor & (self.upsample_factor - 1):
            raise ValueError(
                "HoyeolStyleF0CodecDecoder upsample_factor must be a power of two; "
                f"got {self.upsample_factor}."
            )
        self.num_upsamples = int(math.log2(self.upsample_factor))

        self.input_proj = (
            WNConv1d(self.in_channels, self.dim, kernel_size=1)
            if self.in_channels != self.dim else nn.Identity()
        )

        negative_slope = 0.1
        self.condition_fuser = None
        if self.speaker_condition and self.use_spk_concat:
            self.condition_fuser = nn.Sequential(
                nn.Conv1d(self.dim + self.condition_dim, self.dim, kernel_size=1),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Conv1d(self.dim, self.dim, kernel_size=1),
            )

        self.speaker_stage_film = nn.ModuleList()
        for _ in range(self.num_layers):
            if self.speaker_condition and self.use_spk_film:
                self.speaker_stage_film.append(
                    nn.Sequential(
                        nn.Linear(self.condition_dim, self.dim * 2),
                        nn.LeakyReLU(negative_slope=negative_slope),
                        nn.Linear(self.dim * 2, self.dim * 2),
                    )
                )
            else:
                self.speaker_stage_film.append(nn.Identity())

        def _make_transformer(n_layers: int) -> Transformer:
            return Transformer(
                n_layers=n_layers,
                dim=self.dim,
                num_heads=int(num_heads),
                window_size=None,
                attn_window_size=attn_window_size,
                intermediate_dim=self.intermediate_dim,
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

        self.transformer_backbone_level2 = _make_transformer(self.level2_num_layers)
        self.transformer_backbone_level1 = _make_transformer(self.level1_num_layers)

        if self.use_mhca:
            self.mhca_list = nn.ModuleList(
                [
                    CrossAttentionBlock(
                        query_dim=self.dim,
                        key_dim=mhca_key_dim,
                        num_heads=mhca_num_heads,
                        dropout=mhca_dropout,
                        use_sdpa=mhca_use_sdpa,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            self.mhca_list = None

        self.final_norm = _build_token_norm(self.dim, norm_type=norm_type, norm_eps=norm_eps)
        self.upsamplers = nn.ModuleList(
            [
                ConvUpsample(
                    self.dim,
                    self.dim,
                    factor=2,
                    norm_type=norm_type,
                    norm_eps=norm_eps,
                )
                for _ in range(self.num_upsamples)
            ]
        )
        if self.fcpe_loss_enabled:
            fcpe_head_dim = (
                fcpe_mixture_param_count(self.fcpe_mixture_components)
                if self.fcpe_uses_mixture
                else self.fcpe_out_dims
            )
            self.fcpe_head = WNConv1d(self.dim, fcpe_head_dim, kernel_size=1)
            self.output_head = None
            self.last_fcpe_logits = None
            self.last_fcpe_mixture_params = None
        else:
            self.fcpe_head = None
            self.output_head = WNConv1d(self.dim, self.output_channels, kernel_size=7, padding=3)
            self.last_fcpe_logits = None
            self.last_fcpe_mixture_params = None
        self.last_forward_debug = {}

        stage_dims = [self.dim] * (self.num_layers * self.stage_output_repeat)
        self.width_list = [list(reversed(stage_dims))]

        self.reset_parameters()
        print(
            "HoyeolStyleF0CodecDecoder: "
            f"in_channels={self.in_channels}, dim={self.dim}, output_channels={self.output_channels}, "
            f"level2_num_layers={self.level2_num_layers}, level1_num_layers={self.level1_num_layers}, "
            f"upsample_factor={self.upsample_factor}, num_upsamples={self.num_upsamples}, "
            f"fcpe_loss_mode={self.fcpe_loss_mode}, stage_output_repeat={self.stage_output_repeat}"
        )
        print(
            "HoyeolStyleF0CodecDecoder stack: "
            "input_proj -> transformer_backbone_level2 -> transformer_backbone_level1 "
            f"-> ConvUpsample x{self.num_upsamples} -> "
            f"{'fcpe_head' if self.fcpe_loss_enabled else 'output_head'}"
        )
        if self.speaker_condition:
            print(
                "HoyeolStyleF0CodecDecoder speaker conditioning: "
                f"concat={self.use_spk_concat}, film={self.use_spk_film}, mhca={self.use_mhca}"
            )

    def _iter_transformer_blocks(self):
        yield from self.transformer_backbone_level2.layers
        yield from self.transformer_backbone_level1.layers

    def _apply_global_speaker_condition(self, x: torch.Tensor, spk_emb: Optional[torch.Tensor]) -> torch.Tensor:
        if self.condition_fuser is None or spk_emb is None:
            return x
        spk_time = spk_emb.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return self.condition_fuser(torch.cat([x, spk_time], dim=1))

    def _apply_stage_speaker_film(self, x: torch.Tensor, spk_emb: Optional[torch.Tensor], idx: int) -> torch.Tensor:
        if not (self.speaker_condition and self.use_spk_film) or spk_emb is None:
            return x
        gamma_beta = self.speaker_stage_film[idx](spk_emb)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return x * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

    def _expand_stage_outputs(self, stage_outputs: list[torch.Tensor]) -> list[torch.Tensor]:
        expanded = []
        for tensor in stage_outputs:
            expanded.extend([tensor] * self.stage_output_repeat)
        return expanded

    def forward(self, x, spk_emb=None):
        if self.use_mhca and isinstance(spk_emb, tuple):
            spk_emb_global, mhca_key = spk_emb
        else:
            spk_emb_global = spk_emb
            mhca_key = None

        if self.speaker_condition:
            assert spk_emb_global is not None, "Speaker embedding is required when speaker_condition=True"

        debug_info = {
            "input": _tensor_debug_stats(x),
            "speaker": _tensor_debug_stats(spk_emb_global) if isinstance(spk_emb_global, torch.Tensor) else None,
            "mhca_key": _tensor_debug_stats(mhca_key) if isinstance(mhca_key, torch.Tensor) else None,
        }

        x = self.input_proj(x)
        debug_info["after_input_proj"] = _tensor_debug_stats(x)
        x = self._apply_global_speaker_condition(x, spk_emb_global)

        stage_outputs = []
        for idx, block in enumerate(self._iter_transformer_blocks()):
            if self.use_mhca and mhca_key is not None:
                x = self.mhca_list[idx](x, mhca_key)
                debug_info[f"after_mhca_{idx + 1}"] = _tensor_debug_stats(x)
            x = self._apply_stage_speaker_film(x, spk_emb_global, idx)
            x = block(x)
            debug_info[f"after_transformer_{idx + 1}"] = _tensor_debug_stats(x)
            stage_outputs.append(x)

        outs = self._expand_stage_outputs(stage_outputs)
        x = self.final_norm(x.transpose(1, 2)).transpose(1, 2)
        debug_info["after_final_norm"] = _tensor_debug_stats(x)

        for idx, upsample in enumerate(self.upsamplers):
            x = upsample(x)
            debug_info[f"after_convupsample_{idx + 1}"] = _tensor_debug_stats(x)

        if self.fcpe_loss_enabled:
            with autocast(device_type=x.device.type, enabled=False):
                fcpe_head_out = self.fcpe_head(x.float())
                if self.fcpe_uses_mixture:
                    fcpe_latent = fcpe_mixture_params_to_distribution(
                        fcpe_head_out,
                        num_bins=self.fcpe_out_dims,
                        components=self.fcpe_mixture_components,
                        min_sigma=self.fcpe_mixture_min_sigma,
                        max_sigma=self.fcpe_mixture_max_sigma,
                    )
                    fcpe_logits = fcpe_latent.clamp_min(1.0e-8).log()
                    self.last_fcpe_mixture_params = fcpe_head_out
                else:
                    fcpe_logits = fcpe_head_out
                    fcpe_latent = torch.sigmoid(fcpe_logits)
                    self.last_fcpe_mixture_params = None
            self.last_fcpe_logits = fcpe_logits
            debug_info["fcpe_logits"] = _tensor_debug_stats(fcpe_logits)
            debug_info["fcpe_mixture_params"] = (
                _tensor_debug_stats(self.last_fcpe_mixture_params)
                if self.last_fcpe_mixture_params is not None
                else None
            )
            self.last_forward_debug = debug_info
            return outs, fcpe_latent

        self.last_fcpe_logits = None
        self.last_fcpe_mixture_params = None
        x = self.output_head(x)
        debug_info["final_output"] = _tensor_debug_stats(x)
        self.last_forward_debug = debug_info
        outs.append(x)
        return outs

    def reset_parameters(self):
        def _init(module):
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_init)
