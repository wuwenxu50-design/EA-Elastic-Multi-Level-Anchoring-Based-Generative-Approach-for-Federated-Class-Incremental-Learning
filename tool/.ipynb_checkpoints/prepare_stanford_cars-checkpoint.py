import os
import shutil
from pathlib import Path

import scipy.io as sio


def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path):
    """Prefer symlink; fallback to copy."""
    if dst.exists():
        return
    _safe_mkdir(dst.parent)
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _load_annos(mat_path: Path):
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    annos = mat["annotations"]
    if hasattr(annos, "shape"):
        annos = annos.reshape(-1)
    else:
        annos = [annos]
    return annos


def _get_class_id(a) -> int:
    # IMPORTANT: "class" is a Python keyword, cannot use a.class
    return int(getattr(a, "class")) - 1  # 1-based -> 0-based


def main():
    root = Path.home() / "datasets" / "stanford_cars"
    raw = root / "raw"
    cars_train = raw / "cars_train"
    cars_test = raw / "cars_test"
    devkit = raw / "devkit"

    assert cars_train.exists(), f"Missing {cars_train}"
    assert cars_test.exists(), f"Missing {cars_test}"
    assert devkit.exists(), f"Missing {devkit}"

    meta = sio.loadmat(devkit / "cars_meta.mat", squeeze_me=True, struct_as_record=False)
    class_names = [str(x).strip() for x in meta["class_names"]]

    train_annos = _load_annos(devkit / "cars_train_annos.mat")
    test_annos_
