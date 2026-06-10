"""Data module wrapper for SDPCodec."""

from __future__ import annotations

from ptl.bicodec.data_module import DataModule as _CoreCodecDataModule


class SdpCodecDataModule(_CoreCodecDataModule):
    """LibriSpeech + MLS data pipeline used by the RVQ6561 baseline."""
