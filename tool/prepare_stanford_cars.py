import os
import shutil
from pathlib import Path
import scipy.io as sio


def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    _safe_mkdir(dst.parent)
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _load_annos(mat_path: Path):
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    annos = mat["annotations"]
    if hasattr(annos, "shape"):
        annos = annos.reshape(-1)
    else:
        annos = [annos]
    return annos


def _get_class_id(a) -> int:
    # "class" is a Python keyword; must use getattr
    return int(getattr(a, "class")) - 1


def _pick_file(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("None of these exists:\n" + "\n".join(str(p) for p in candidates))


def main():
    root = Path("/root/datasets/stanford_cars")
    raw = root / "raw"

    cars_train = raw / "cars_train"
    cars_test = raw / "cars_test"
    devkit = raw / "devkit"
    alt = raw / "stanford_cars"  # some packs put mats here

    print("[INFO] root =", root)
    print("[INFO] raw  =", raw)
    print("[INFO] cars_train =", cars_train, "exists:", cars_train.exists())
    print("[INFO] cars_test  =", cars_test, "exists:", cars_test.exists())
    print("[INFO] devkit     =", devkit, "exists:", devkit.exists())
    print("[INFO] alt        =", alt, "exists:", alt.exists())

    assert cars_train.exists(), f"Missing {cars_train}"
    assert cars_test.exists(), f"Missing {cars_test}"

    cars_meta = _pick_file(devkit / "cars_meta.mat", alt / "cars_meta.mat")
    train_ann = _pick_file(devkit / "cars_train_annos.mat", alt / "cars_train_annos.mat")
    test_ann  = _pick_file(devkit / "cars_test_annos_withlabels.mat", alt / "cars_test_annos_withlabels.mat")

    print("[INFO] cars_meta =", cars_meta)
    print("[INFO] train_ann =", train_ann)
    print("[INFO] test_ann  =", test_ann)

    meta = sio.loadmat(str(cars_meta), squeeze_me=True, struct_as_record=False)
    class_names = [str(x).strip() for x in meta["class_names"]]
    print("[INFO] num classes =", len(class_names))

    train_annos = _load_annos(train_ann)
    test_annos = _load_annos(test_ann)
    print("[INFO] train annos =", len(train_annos))
    print("[INFO] test  annos =", len(test_annos))

    out_train = root / "train"
    out_test = root / "test"
    _safe_mkdir(out_train)
    _safe_mkdir(out_test)

    for a in train_annos:
        fname = str(getattr(a, "fname")).strip()
        cls = _get_class_id(a)
        cls_name = class_names[cls].replace("/", "_")
        cls_dir = out_train / f"{cls:03d}_{cls_name}"
        link_or_copy(cars_train / fname, cls_dir / fname)

    for a in test_annos:
        fname = str(getattr(a, "fname")).strip()
        cls = _get_class_id(a)
        cls_name = class_names[cls].replace("/", "_")
        cls_dir = out_test / f"{cls:03d}_{cls_name}"
        link_or_copy(cars_test / fname, cls_dir / fname)

    n_train_cls = len([p for p in out_train.iterdir() if p.is_dir()])
    n_test_cls = len([p for p in out_test.iterdir() if p.is_dir()])
    print("[DONE] Train root:", out_train, "classes:", n_train_cls)
    print("[DONE] Test  root:", out_test, "classes:", n_test_cls)


if __name__ == "__main__":
    main()
