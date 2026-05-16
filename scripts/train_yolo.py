"""Treina YOLOv8n no dataset CholecSeg8k convertido (formato YOLO).

Alternativa local ao notebook `notebooks/train_yolo_colab.ipynb` (caminho
recomendado para reproduzir o treino no Colab). Use este script quando o
dataset já estiver convertido em `data/processed/cholecseg8k_yolo/` e você
quiser rodar localmente em GPU CUDA, MPS (Apple Silicon) ou CPU.

Uso típico:
    python scripts/train_yolo.py \\
        --data data/processed/cholecseg8k_yolo/data.yaml \\
        --epochs 40

Saída em `runs/detect/<name>/`:
    - `weights/best.pt`: pesos finais (copiar para `models/yolov8n_surgical.pt`)
    - `results.csv`: métricas por epoch
    - `results.png`: curvas de loss/mAP
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/cholecseg8k_yolo/data.yaml"),
        help="Path do data.yaml gerado por convert_cholecseg8k_to_yolo.py",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Pesos base (Ultralytics baixa automaticamente se não existir)",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--name",
        type=str,
        default="surgical_instruments",
        help="Subpasta de runs/detect/ onde resultados serão salvos",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="'0' p/ GPU CUDA 0, 'mps' p/ Apple Silicon, 'cpu' p/ forçar CPU. "
        "Vazio = autodetect.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping: epochs sem melhora antes de parar",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Pula validação final (mantém só métricas durante o treino)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"data.yaml não encontrado em {args.data}. "
            "Rode `python scripts/convert_cholecseg8k_to_yolo.py` antes."
        )

    from ultralytics import YOLO

    print(f"[train_yolo] dataset: {args.data}")
    print(f"[train_yolo] base weights: {args.weights}")
    print(
        f"[train_yolo] epochs={args.epochs} imgsz={args.imgsz} "
        f"batch={args.batch} device={args.device or 'auto'}"
    )

    model = YOLO(args.weights)

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        name=args.name,
        patience=args.patience,
        exist_ok=True,
        verbose=True,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None
    if save_dir:
        print(f"\n[train_yolo] resultados em: {save_dir}")
        best = save_dir / "weights" / "best.pt"
        if best.exists():
            size_mb = best.stat().st_size / (1024 * 1024)
            print(f"[train_yolo] best.pt: {best} ({size_mb:.1f} MB)")

    if not args.no_val:
        print("\n[train_yolo] rodando validação final no split test...")
        metrics = model.val(data=str(args.data), split="test")
        print(f"[train_yolo] mAP50 (test):    {metrics.box.map50:.4f}")
        print(f"[train_yolo] mAP50-95 (test): {metrics.box.map:.4f}")
        print(f"[train_yolo] precision:       {metrics.box.mp:.4f}")
        print(f"[train_yolo] recall:          {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
