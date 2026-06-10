"""Translate the clean SDPCodec config surface to legacy implementation keys."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf, open_dict


def _get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [] if value.lower() in {"", "none", "null"} else [value]
    return list(value)


def _ensure_child(node: Any, key: str) -> Any:
    child = _get(node, key, None)
    if child is None or not (isinstance(child, dict) or OmegaConf.is_dict(child)):
        node[key] = {}
        child = node[key]
    return child


def _set_path(cfg: Any, dotted_path: str, value: Any) -> None:
    cur = cfg
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        cur = _ensure_child(cur, part)
    cur[parts[-1]] = value


def expand_legacy_config(cfg: Any) -> Any:
    """Add compatibility keys expected by the copied BigCodec modules.

    The checked-in YAML stays selector/list based. This function mutates only the
    in-memory config passed to the legacy dataloader and Lightning module.
    """

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(cfg):
        _expand_f0_codec(cfg)
        _expand_speaker_encoder(cfg)
        _expand_codec_encoder(cfg)
        _expand_codec_decoder(cfg)
        _expand_discriminators(cfg)
        _expand_train(cfg)
    return cfg


def _expand_f0_codec(cfg: Any) -> None:
    f0 = cfg.model.f0_codec
    losses = {str(item).lower() for item in _as_list(_get(f0, "losses", []))}
    encoder = _get(f0, "encoder", {})
    decoder = _get(f0, "decoder", {})
    decoder_conditioning = [str(item).lower() for item in _as_list(_get(decoder, "speaker_conditioning", []))]

    _set_path(cfg, "model.f0_codec.use_codec_structure", str(_get(f0, "structure", "codec")).lower() == "codec")
    _set_path(cfg, "model.f0_codec.use_normalized_f0", str(_get(f0, "pitch_representation", "normalized")).lower() == "normalized")
    _set_path(cfg, "model.f0_codec.use_fcpe_loss", "fcpe" in losses)
    _set_path(cfg, "model.f0_codec.use_unnormf0_mse_loss", "unnorm_mse" in losses)
    _set_path(cfg, "model.f0_codec.upsample_extracted_f0", str(_get(f0, "extraction_rate", "frame")).lower() == "audio")

    enc_rnn = str(_get(encoder, "recurrent", "lstm")).lower()
    dec_rnn = str(_get(decoder, "recurrent", "lstm")).lower()
    _set_path(cfg, "model.f0_codec.encoder_use_rnn", enc_rnn not in {"none", "null", ""})
    _set_path(cfg, "model.f0_codec.encoder_rnn_type", enc_rnn)
    _set_path(cfg, "model.f0_codec.encoder_rnn_num_layers", int(_get(encoder, "recurrent_layers", 2)))
    _set_path(cfg, "model.f0_codec.encoder_ngf", int(_get(encoder, "ngf", _get(f0, "encoder_ngf", 32))))
    _set_path(cfg, "model.f0_codec.encoder_up_ratios", _as_list(_get(encoder, "up_ratios", [1, 1, 2, 2])))
    _set_path(cfg, "model.f0_codec.encoder_dilations", _as_list(_get(encoder, "dilations", [1, 3, 9])))
    _set_path(cfg, "model.f0_codec.encoder_out_channels", int(_get(encoder, "out_channels", 128)))
    _set_path(cfg, "model.f0_codec.encoder_activation_type", _get(encoder, "activation", "LeakyReLU"))
    _set_path(cfg, "model.f0_codec.encoder_leaky_relu_params.negative_slope", float(_get(encoder, "negative_slope", 0.1)))

    _set_path(cfg, "model.f0_codec.decoder_use_rnn", dec_rnn not in {"none", "null", ""})
    _set_path(cfg, "model.f0_codec.decoder_rnn_type", dec_rnn)
    _set_path(cfg, "model.f0_codec.decoder_rnn_num_layers", int(_get(decoder, "recurrent_layers", 2)))
    _set_path(cfg, "model.f0_codec.decoder_ngf", int(_get(decoder, "ngf", _get(f0, "decoder_ngf", 32))))
    _set_path(cfg, "model.f0_codec.decoder_up_ratios", _as_list(_get(decoder, "up_ratios", [2, 2, 1, 1])))
    _set_path(cfg, "model.f0_codec.decoder_dilations", _as_list(_get(decoder, "dilations", [1, 3, 9])))
    _set_path(cfg, "model.f0_codec.decoder_activation_type", _get(decoder, "activation", "LeakyReLU"))
    _set_path(cfg, "model.f0_codec.decoder_leaky_relu_params.negative_slope", float(_get(decoder, "negative_slope", 0.1)))
    _set_path(cfg, "model.f0_codec.decoder_use_mhca", "mhca" in decoder_conditioning)
    _set_path(cfg, "model.f0_codec.decoder_mhca_num_heads", int(_get(decoder, "mhca_heads", 2)))
    _set_path(cfg, "model.f0_codec.decoder_mhca_dropout", float(_get(decoder, "mhca_dropout", 0.2)))
    _set_path(cfg, "model.f0_codec.decoder_mhca_key_dim", int(_get(decoder, "mhca_key_dim", 128)))
    _set_path(cfg, "model.f0_codec.decoder_mhca_use_sdpa", str(_get(decoder, "attention_backend", "sdpa")).lower() == "sdpa")
    _set_path(cfg, "model.f0_codec.spk_cond.type", decoder_conditioning)
    _set_path(cfg, "model.f0_codec.spk_cond.num_heads", cfg.model.f0_codec.decoder_mhca_num_heads)
    _set_path(cfg, "model.f0_codec.spk_cond.dropout", cfg.model.f0_codec.decoder_mhca_dropout)
    _set_path(cfg, "model.f0_codec.spk_cond.key_dim", cfg.model.f0_codec.decoder_mhca_key_dim)
    _set_path(cfg, "model.f0_codec.spk_cond.use_sdpa", cfg.model.f0_codec.decoder_mhca_use_sdpa)


def _expand_speaker_encoder(cfg: Any) -> None:
    spk = cfg.model.speaker_encoder
    quantizer = str(_get(spk, "quantizer", "none")).lower()
    post = _get(spk, "post_encoder", {})
    stages = _as_list(_get(post, "stages", []))
    first_stage = stages[0] if stages else {}
    first_type = str(_get(first_stage, "type", "none")).lower()

    _set_path(cfg, "model.speaker_encoder.speaker_encoder_type", _get(spk, "encoder", "wavlm"))
    _set_path(cfg, "model.speaker_encoder.use_quantizer", quantizer != "none")
    _set_path(cfg, "model.speaker_encoder.quantizer.type", quantizer)
    _set_path(cfg, "model.speaker_encoder.use_perceiver_encoder", first_type in {"perceiver", "memory_cross_attention"})
    _set_path(cfg, "model.speaker_encoder.use_memory_cattn", first_type == "memory_cross_attention")
    _set_path(cfg, "model.speaker_encoder.perceiver_use_flash_attn", str(_get(first_stage, "attention_backend", "flash")).lower() == "flash")
    _set_path(cfg, "model.speaker_encoder.token_num", int(_get(first_stage, "tokens", _get(spk, "token_num", 128))))
    _set_path(cfg, "model.speaker_encoder.latent_dim", int(_get(first_stage, "dim", _get(spk, "latent_dim", 128))))

    if len(stages) > 1:
        stage_map = {
            "memory_cross_attention": "mca",
            "perceiver": "pe",
            "token_mixer": "tm",
        }
        _set_path(cfg, "model.speaker_encoder.stack.use", True)
        _set_path(cfg, "model.speaker_encoder.stack.layers", [stage_map.get(str(_get(stage, "type")).lower(), _get(stage, "type")) for stage in stages])
        _set_path(cfg, "model.speaker_encoder.stack.token_nums", [_get(stage, "tokens", None) for stage in stages])
        _set_path(cfg, "model.speaker_encoder.stack.latent_dims", [_get(stage, "dim", None) for stage in stages])


def _expand_codec_encoder(cfg: Any) -> None:
    enc = cfg.model.codec_encoder
    encoder = str(_get(enc, "encoder", "hubert")).lower()
    feature = str(_get(enc, "feature", "transformer")).lower()
    trainable = bool(_get(enc, "trainable", False))

    for name in ("vqw2v", "w2v2", "hubert", "w2vbert2", "s3tokenizer"):
        _set_path(cfg, f"model.codec_encoder.use_{name}", encoder == name)
        _set_path(cfg, f"model.codec_encoder.freeze_{name}_encoder", not trainable)

    _set_path(cfg, "model.codec_encoder.hubert_use_transformer", feature == "transformer")
    _set_path(cfg, "model.codec_encoder.use_vqw2v_continuous", str(_get(enc, "representation", "continuous")).lower() == "continuous")
    temporal = _get(enc, "temporal", "none")
    if isinstance(temporal, str):
        _set_path(cfg, "model.codec_encoder.temporal.use", temporal.lower() not in {"none", "null", ""})
        _set_path(cfg, "model.codec_encoder.temporal.type", "lstm" if temporal.lower() in {"none", "null", ""} else temporal)
        _set_path(cfg, "model.codec_encoder.temporal.bidirectional", False)
        _set_path(cfg, "model.codec_encoder.temporal.mamba", {})


def _expand_codec_decoder(cfg: Any) -> None:
    dec = cfg.model.codec_decoder
    speaker_conditioning = [str(item).lower() for item in _as_list(_get(dec, "speaker_conditioning", []))]
    f0_conditioning = [str(item).lower() for item in _as_list(_get(dec, "f0_conditioning", ["concat"]))]
    quantizer = str(_get(dec, "quantizer", "rvq")).lower()

    _set_path(cfg, "model.codec_decoder.quantizer_type", quantizer)
    _set_path(cfg, "model.codec_decoder.vq_num_quantizers", int(_get(dec, "quantizers", _get(dec, "vq_num_quantizers", 1))))
    _set_path(cfg, "model.codec_decoder.use_vqw2v_embed", False)
    _set_path(cfg, "model.codec_decoder.speaker_condition", len(speaker_conditioning) > 0 and "none" not in speaker_conditioning)
    _set_path(cfg, "model.codec_decoder.f0_condition", len(f0_conditioning) > 0 and "none" not in f0_conditioning)
    _set_path(cfg, "model.codec_decoder.f0_speaker_condition", str(_get(dec, "f0_speaker_conditioning", "enabled")).lower() not in {"none", "disabled", "false"})
    _set_path(cfg, "model.codec_decoder.use_stage_speaker_film", "stage_film" in speaker_conditioning)
    _set_path(cfg, "model.codec_decoder.use_mhca", "mhca" in speaker_conditioning)
    _set_path(cfg, "model.codec_decoder.spk_cond.use", len(speaker_conditioning) > 0 and "none" not in speaker_conditioning)
    _set_path(cfg, "model.codec_decoder.spk_cond.type", speaker_conditioning)
    _set_path(cfg, "model.codec_decoder.spk_cond.num_heads", int(_get(dec, "mhca_heads", 2)))
    _set_path(cfg, "model.codec_decoder.spk_cond.dropout", float(_get(dec, "mhca_dropout", 0.2)))
    _set_path(cfg, "model.codec_decoder.mhca_num_heads", int(_get(dec, "mhca_heads", 2)))
    _set_path(cfg, "model.codec_decoder.mhca_dropout", float(_get(dec, "mhca_dropout", 0.2)))
    _set_path(cfg, "model.codec_decoder.mhca_use_sdpa", str(_get(dec, "mhca_attention_backend", "sdpa")).lower() == "sdpa")

    temporal = _get(dec, "temporal", {"type": "lstm"})
    if isinstance(temporal, str):
        _set_path(cfg, "model.codec_decoder.temporal.use", temporal.lower() not in {"none", "null", ""})
        _set_path(cfg, "model.codec_decoder.temporal.type", "lstm" if temporal.lower() in {"none", "null", ""} else temporal)
    else:
        temporal_type = str(_get(temporal, "type", "lstm")).lower()
        _set_path(cfg, "model.codec_decoder.temporal.use", temporal_type not in {"none", "null", ""})
        _set_path(cfg, "model.codec_decoder.temporal.type", temporal_type)
    _set_path(cfg, "model.codec_decoder.temporal.bidirectional", False)
    _set_path(cfg, "model.codec_decoder.temporal.mamba", {})


def _expand_discriminators(cfg: Any) -> None:
    mstft = cfg.model.mstft
    _set_path(cfg, "model.mstft.use_weight_norm", bool(_get(mstft, "weight_norm", True)))


def _expand_train(cfg: Any) -> None:
    train = cfg.train
    losses = {str(item).lower() for item in _as_list(_get(train, "losses", []))}
    validation_metrics = {str(item).lower() for item in _as_list(_get(train, "validation_metrics", []))}
    compile_mode = str(_get(train, "compile", "none")).lower()

    _set_path(cfg, "train.use_mel_loss", "mel" in losses)
    _set_path(cfg, "train.use_feat_match_loss", "feat_match" in losses)
    _set_path(cfg, "train.gan_loss.use_bigvsan", str(_get(train.gan_loss, "mode", "lsgan")).lower() == "bigvsan")
    _set_path(cfg, "train.use_torch_compile", compile_mode not in {"none", "off", "false"})
    _set_path(cfg, "train.compile_mode", "default" if compile_mode in {"none", "off", "false"} else compile_mode)
    _set_path(cfg, "train.use_val_utmos", "utmos" in validation_metrics)
    _set_path(cfg, "train.use_val_wavlm", "wavlm" in validation_metrics)
    _set_path(cfg, "train.use_val_wer", "wer" in validation_metrics)
    _set_path(cfg, "train.apn_aux_loss.use", _get(train.apn_aux_loss, "enabled", None))
