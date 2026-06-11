import os
import shutil
from contextlib import contextmanager
from typing import Callable, Optional, Union

import torch
import torch.distributed as dist


def _default_torch_home() -> str:
    if os.path.isdir("/data/hounsu"):
        return "/data/hounsu/.cache/torch"
    if os.path.isdir("/mnt/HDD/hounsu"):
        return "/mnt/HDD/hounsu/.cache/torch"

    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        return os.path.join(xdg_cache, "torch")
    return os.path.expanduser("~/.cache/torch")


def configure_torch_hub_cache() -> str:
    """Use a stable torch.hub cache outside the repo by default."""
    torch_home = os.environ.get("TORCH_HOME", "").strip()
    repo_root = os.environ.get("HOME_BIGCODEC_ROOT", "/home/hounsu/voice/bigcodec")
    allow_home_cache = os.environ.get("BIGCODEC_ALLOW_HOME_TORCH_CACHE", "0") == "1"
    default_torch_home = _default_torch_home()

    if not torch_home:
        torch_home = default_torch_home

    if (
        not allow_home_cache
        and os.path.abspath(torch_home).startswith(os.path.abspath(repo_root))
    ):
        torch_home = default_torch_home

    os.makedirs(torch_home, exist_ok=True)
    os.environ["TORCH_HOME"] = torch_home
    torch.hub.set_dir(torch_home)
    return torch_home


def configure_s3prl_cache_root(torch_home: Optional[str] = None) -> str:
    """Use an S3PRL cache outside the repo by default."""
    s3prl_cache_root = os.environ.get("S3PRL_CACHE_ROOT", "").strip()
    repo_root = os.environ.get("HOME_BIGCODEC_ROOT", "/home/hounsu/voice/bigcodec")
    allow_home_cache = os.environ.get("BIGCODEC_ALLOW_HOME_TORCH_CACHE", "0") == "1"
    default_root = os.path.join(torch_home or _default_torch_home(), "s3prl")

    if not s3prl_cache_root:
        s3prl_cache_root = default_root

    if (
        not allow_home_cache
        and os.path.abspath(s3prl_cache_root).startswith(os.path.abspath(repo_root))
    ):
        s3prl_cache_root = default_root

    os.makedirs(s3prl_cache_root, exist_ok=True)
    os.environ["S3PRL_CACHE_ROOT"] = s3prl_cache_root
    return s3prl_cache_root


def resolve_wavlm_sv_checkpoint_path(
    explicit_path: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> str:
    """Resolve the WavLM speaker-verification checkpoint used for validation."""
    rel_path = os.path.join("etc", "unispeech_models", "wavlm_large_finetune.pth")
    candidates = []

    for raw_path in (
        explicit_path,
        os.environ.get("BIGCODEC_WAVLM_SV_CKPT"),
        os.environ.get("VAL_WAVLM_CKPT_PATH"),
    ):
        if raw_path:
            candidates.append(os.path.expanduser(str(raw_path)))

    if repo_root:
        candidates.append(os.path.join(str(repo_root), rel_path))

    home_repo = os.environ.get("HOME_BIGCODEC_ROOT", "").strip()
    if home_repo:
        candidates.append(os.path.join(home_repo, rel_path))

    candidates.append(os.path.join("/home/hounsu/voice/bigcodec", rel_path))

    seen = set()
    checked = []
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        checked.append(candidate)
        if os.path.isfile(candidate):
            return candidate

    checked_text = "\n  - ".join(checked)
    raise FileNotFoundError(
        "WavLM speaker-verification checkpoint not found. Checked:\n"
        f"  - {checked_text}\n"
        "Set BIGCODEC_WAVLM_SV_CKPT or train.val_wavlm_checkpoint_path to the exact file."
    )


SPEECHMOS_REPO_CACHE_NAME = "tarepan_SpeechMOS_v1.2.0"


def speechmos_repo_cache_dir(torch_home: str) -> str:
    repo_dirs = [
        os.path.join(torch_home, SPEECHMOS_REPO_CACHE_NAME),
        os.path.join(torch_home, "hub", SPEECHMOS_REPO_CACHE_NAME),
    ]
    for repo_dir in repo_dirs:
        if os.path.isdir(repo_dir):
            return repo_dir
    return repo_dirs[0]


def speechmos_hubconf_path(torch_home: str) -> str:
    return os.path.join(speechmos_repo_cache_dir(torch_home), "hubconf.py")


def _hubconf_path(repo_dir: str) -> str:
    return os.path.join(repo_dir, "hubconf.py")


def remove_incomplete_speechmos_cache(torch_home: str) -> bool:
    """Remove a corrupt torch.hub SpeechMOS repo cache.

    torch.hub treats the repo directory itself as a cache hit. If another run or
    interrupted download leaves the directory without hubconf.py, every later
    validation crashes with FileNotFoundError unless we clear the partial cache.
    """
    repo_dir = speechmos_repo_cache_dir(torch_home)
    if not os.path.isdir(repo_dir) or os.path.isfile(speechmos_hubconf_path(torch_home)):
        return False
    shutil.rmtree(repo_dir, ignore_errors=True)
    return True


def ensure_private_speechmos_repo(torch_home: str, private_root: str) -> Optional[str]:
    """Copy the SpeechMOS torch.hub repo into a run-local immutable location.

    The model object can still be deleted after validation. The important part is
    that later validations import from this private repo copy instead of touching
    the shared torch.hub cache again.
    """
    private_repo = os.path.join(private_root, SPEECHMOS_REPO_CACHE_NAME)
    if os.path.isfile(_hubconf_path(private_repo)):
        return private_repo
    if os.path.isdir(private_repo):
        shutil.rmtree(private_repo, ignore_errors=True)

    shared_repo = speechmos_repo_cache_dir(torch_home)
    if not os.path.isfile(_hubconf_path(shared_repo)):
        return None

    os.makedirs(private_root, exist_ok=True)
    shutil.copytree(shared_repo, private_repo)
    return private_repo


@contextmanager
def torch_hub_cache_lock(torch_home: str, name: str = "torch_hub"):
    """Serialize torch.hub cache mutation across local training processes."""
    os.makedirs(torch_home, exist_ok=True)
    lock_path = os.path.join(torch_home, f".{name}.lock")
    lock_file = open(lock_path, "a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_file.close()


def dist_control_device(preferred_device: Optional[Union[torch.device, str]] = None) -> torch.device:
    if not (dist.is_available() and dist.is_initialized()):
        return torch.device("cpu")
    backend = str(dist.get_backend()).lower()
    if backend == "nccl":
        if preferred_device is not None:
            device = torch.device(preferred_device)
            if device.type == "cuda":
                return device
        local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def run_rank_ordered_or_raise(load_fn: Callable[[], None], tag: str, control_device: torch.device) -> None:
    """Run load_fn one rank at a time and propagate Python exceptions to all ranks.

    Without this, one rank can raise while another rank waits in a barrier or a
    sync_dist metric reduction until NCCL's watchdog timeout fires much later.
    """
    if not (dist.is_available() and dist.is_initialized()):
        load_fn()
        return

    rank = dist.get_rank()
    world = dist.get_world_size()

    for load_rank in range(world):
        failed = False
        if rank == load_rank:
            try:
                load_fn()
            except Exception as exc:
                failed = True
                print(f"[{tag}] rank {rank}/{world} failed: {type(exc).__name__}: {exc}", flush=True)

        flag = torch.tensor([1 if failed else 0], device=control_device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if int(flag.item()) != 0:
            raise RuntimeError(f"{tag} failed on at least one rank; aborting all DDP ranks before NCCL timeout.")
        dist.barrier()
