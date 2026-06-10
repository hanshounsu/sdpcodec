"""Lightning module wrapper for the clean SDPCodec API."""

from __future__ import annotations

from sdpcodec._compat import ensure_legacy_import_paths

ensure_legacy_import_paths()

from ptl.trixcodec.lightning_module import TriXCodecLightningModule


class SdpCodecLightningModule(TriXCodecLightningModule):
    """Joint content/F0 codec system used by the SDPCodec baseline."""


SdpCodecSystem = SdpCodecLightningModule
