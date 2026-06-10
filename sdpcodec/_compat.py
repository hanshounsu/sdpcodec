"""Import-path bridge for the legacy modules copied from BigCodec."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_legacy_import_paths() -> None:
    """Expose copied legacy top-level modules exactly as the old repo did."""

    for path in (REPO_ROOT, REPO_ROOT / "vq"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
