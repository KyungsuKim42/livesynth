"""Weight download and caching (Hugging Face Hub)."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_REPO = "KyungsuKim/LiveSynth"
MODEL_FILES = ("config.json", "generator.safetensors", "decoder.safetensors",
               "text_align.safetensors", "presets.safetensors")
CLAP_REPO = "lukewys/laion_clap"
CLAP_FILE = "music_audioset_epoch_15_esc_90.14.pt"


def resolve_model_dir(repo_id: str = DEFAULT_REPO, revision: str | None = None,
                      local_dir: str | os.PathLike | None = None,
                      cache_dir: str | os.PathLike | None = None) -> Path:
    """Return a directory containing :data:`MODEL_FILES`.

    Priority: ``local_dir`` argument, then the ``LIVESYNTH_WEIGHTS`` environment
    variable, then a download from ``repo_id`` into the Hugging Face cache (done
    once; later calls reuse the cache). Private repositories need a token
    (``huggingface-cli login`` or ``HF_TOKEN``).
    """
    local = local_dir or os.environ.get("LIVESYNTH_WEIGHTS")
    if local:
        d = Path(local).expanduser()
        missing = [f for f in MODEL_FILES if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(f"{d} is missing {missing}")
        return d
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id, revision=revision, cache_dir=cache_dir,
                                  allow_patterns=list(MODEL_FILES)))


def resolve_clap_checkpoint(cache_dir: str | os.PathLike | None = None) -> str:
    local = os.environ.get("LIVESYNTH_CLAP")
    if local:
        return str(Path(local).expanduser())
    from huggingface_hub import hf_hub_download
    return hf_hub_download(CLAP_REPO, CLAP_FILE, cache_dir=cache_dir)
