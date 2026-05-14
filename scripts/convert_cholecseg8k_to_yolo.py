"""Converte CholecSeg8k para formato YOLOv8 (bbox + splits).

Foco em instrumentos cirúrgicos (alvo 1 do enunciado do Tech Challenge):
- 0 = grasper
- 1 = l_hook_electrocautery

Input:  data/raw/cholecseg8k/ (estrutura: pastas video_<id>/ com frames + masks)
Output: data/processed/cholecseg8k_yolo/
        ├── images/{train,val,test}/*.png
        ├── labels/{train,val,test}/*.txt
        └── data.yaml

Uso:
    python scripts/convert_cholecseg8k_to_yolo.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

RAW_DIR = Path("data/raw/cholecseg8k/extracted/CholecSeg8k")
OUT_DIR = Path("data/processed/cholecseg8k_yolo")

# Classes alvo do YOLO custom (apenas instrumentos cirúrgicos)
INSTRUMENT_CLASSES: dict[str, int] = {
    "grasper": 0,
    "l_hook_electrocautery": 1,
}

# Valores de pixel nas annotation masks (escala de cinza) do CholecSeg8k.
# Fonte: arXiv 2012.12463 (Hong et al, 2020) - 13 classes total.
# Pixel 50 = background; 11=Abdominal Wall; 12=Fat; 13=GI Tract; 21=Liver;
# 22=Gallbladder; 23=Connective Tissue; 24=Blood; 25=Cystic Duct;
# 31=Grasper; 32=L-hook Electrocautery; 33=Hepatic Vein; 5=Liver Ligament.
CLASS_PIXEL_VALUE: dict[str, int] = {
    "grasper": 31,
    "l_hook_electrocautery": 32,
}

SPLIT_RATIO = (0.7, 0.2, 0.1)
SEED = 42
MIN_BBOX_AREA_PX = 100


def mask_to_bboxes_by_class(mask_path: Path) -> dict[str, tuple[int, int, int, int]]:
    """Extrai bbox por classe encontrada na annotation mask em escala de cinza."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return {}
    bboxes: dict[str, tuple[int, int, int, int]] = {}
    for class_name, pixel_val in CLASS_PIXEL_VALUE.items():
        ys, xs = np.where(mask == pixel_val)
        if len(xs) == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        w, h = x_max - x_min, y_max - y_min
        if w * h < MIN_BBOX_AREA_PX:
            continue
        bboxes[class_name] = (x_min, y_min, w, h)
    return bboxes


def yolo_format(bbox: tuple[int, int, int, int], img_w: int, img_h: int, class_id: int) -> str:
    """Formata bbox absoluto como linha YOLO normalizada."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"


def collect_items() -> list[tuple[Path, Path]]:
    """Coleta tuplas (image_path, annotation_mask_path) do dataset.

    Layout do CholecSeg8k (HF minwoosun, descompactado):
        CholecSeg8k/<video_id>/<video_id>_<sub_id>/frame_N_endo.png       (imagem)
        CholecSeg8k/<video_id>/<video_id>_<sub_id>/frame_N_endo_mask.png  (annotation grayscale)
        CholecSeg8k/<video_id>/<video_id>_<sub_id>/frame_N_endo_color_mask.png  (ignorar)
        CholecSeg8k/<video_id>/<video_id>_<sub_id>/frame_N_endo_watershed_mask.png  (ignorar)
    """
    items: list[tuple[Path, Path]] = []
    for img_path in sorted(RAW_DIR.rglob("frame_*_endo.png")):
        # Skip qualquer arquivo que ja seja uma mask (defensive)
        if any(s in img_path.name for s in ("color_mask", "watershed_mask")):
            continue
        # Mask annotation correspondente
        ann_path = img_path.with_name(img_path.name.replace("_endo.png", "_endo_mask.png"))
        if ann_path.exists():
            items.append((img_path, ann_path))
    return items


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    items = collect_items()
    if not items:
        print("ERRO: nenhum par (image, mask) encontrado.")
        print(f"      Conferir RAW_DIR={RAW_DIR} e layout do CholecSeg8k.")
        print("      Adapte collect_items() se o naming for diferente.")
        return

    rng = np.random.default_rng(seed=SEED)
    indices = list(range(len(items)))
    rng.shuffle(indices)

    n = len(items)
    n_train = int(n * SPLIT_RATIO[0])
    n_val = int(n * SPLIT_RATIO[1])
    splits = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    cls_count: dict[int, int] = {cid: 0 for cid in INSTRUMENT_CLASSES.values()}

    for split_name, idxs in splits.items():
        for i in idxs:
            img_path, ann_path = items[i]
            # Nome único: <subpasta>_<filename> pra evitar colisão entre videos/subfolders
            unique_name = f"{img_path.parent.name}_{img_path.name}"
            unique_stem = f"{img_path.parent.name}_{img_path.stem}"
            dst_img = OUT_DIR / "images" / split_name / unique_name
            shutil.copy(img_path, dst_img)
            dst_lbl = OUT_DIR / "labels" / split_name / f"{unique_stem}.txt"

            img = cv2.imread(str(img_path))
            if img is None:
                dst_lbl.touch()
                continue
            h, w = img.shape[:2]

            bboxes = mask_to_bboxes_by_class(ann_path)
            lines: list[str] = []
            for class_name, bbox in bboxes.items():
                class_id = INSTRUMENT_CLASSES[class_name]
                lines.append(yolo_format(bbox, w, h, class_id))
                cls_count[class_id] += 1

            dst_lbl.write_text("".join(lines) if lines else "")

    names_yaml = "\n".join(
        f"  {cid}: {name}" for name, cid in sorted(INSTRUMENT_CLASSES.items(), key=lambda kv: kv[1])
    )
    yaml_text = f"""# CholecSeg8k (instrumentos cirurgicos) - YOLO format
# Source: Hong et al, arXiv 2012.12453, CholecSeg8k 2020
# License: CC BY-NC-SA 4.0
# Gerado por scripts/convert_cholecseg8k_to_yolo.py

path: ../../processed/cholecseg8k_yolo
train: images/train
val: images/val
test: images/test

nc: {len(INSTRUMENT_CLASSES)}
names:
{names_yaml}
"""
    (OUT_DIR / "data.yaml").write_text(yaml_text)

    print(f"OK - {n} frames processados")
    for name, cid in sorted(INSTRUMENT_CLASSES.items(), key=lambda kv: kv[1]):
        print(f"  {name} (id {cid}): {cls_count[cid]} bboxes")
    print(
        f"Splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
    )
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
