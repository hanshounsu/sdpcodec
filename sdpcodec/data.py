"""Data module wrapper for the clean SDPCodec API."""

from __future__ import annotations

from sdpcodec._compat import ensure_legacy_import_paths

ensure_legacy_import_paths()

from ptl.bicodec.data_module import DataModule as _LegacyCodecDataModule


class SdpCodecDataModule(_LegacyCodecDataModule):
    """LibriSpeech + MLS data pipeline used by the RVQ6561 baseline."""
