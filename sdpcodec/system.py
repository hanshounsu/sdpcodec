"""Lightning module wrapper for the clean SDPCodec API."""

from __future__ import annotations

from sdpcodec._compat import ensure_legacy_import_paths

ensure_legacy_import_paths()

from ptl.sdpcodec.lightning_module import SdpCodecLightningModule as _CoreSdpCodecLightningModule


class SdpCodecLightningModule(_CoreSdpCodecLightningModule):
    """Joint content/F0 codec system used by the SDPCodec baseline."""


SdpCodecSystem = SdpCodecLightningModule
