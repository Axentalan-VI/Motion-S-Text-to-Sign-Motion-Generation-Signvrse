"""Mirror checkpoints to a backup directory (typically Google Drive on Colab).

Use from training scripts to survive runtime termination:

    from src.drive_sync import mirror_to_drive
    mirror_to_drive(ckpt_path, drive_dir)
"""
from __future__ import annotations

import shutil
from pathlib import Path


def mirror_to_drive(src: str | Path, drive_dir: str | Path | None) -> None:
    """Copy `src` (a file) into `drive_dir`. No-op if drive_dir is None/empty.

    Errors are caught and logged so a stale Drive mount never kills training.
    """
    if not drive_dir:
        return
    src = Path(src)
    dst_dir = Path(drive_dir)
    if not src.exists():
        print(f"[drive] skip: source missing: {src}")
        return
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        print(f"[drive] mirrored -> {dst}")
    except Exception as e:  # noqa: BLE001
        print(f"[drive] WARN: failed to copy {src} -> {dst_dir}: {e}")
