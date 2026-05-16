"""Empacota frames PNG do CholecSeg8k em MP4s curtos para a aba Multimodal.

Lista as sub-pastas de um `<video_id>` no dataset CholecSeg8k extraido
(`data/raw/cholecseg8k/extracted/CholecSeg8k/<video_id>/<video_id_NN>/`),
seleciona frames `*_endo.png` consecutivos (ignorando `*_mask*.png`),
copia para um diretorio temporario com nome sequencial e dispara `ffmpeg`
para gerar `data/examples/<caso>/video.mp4`.

Dois casos sao gerados:
- `caso_normal`: ~15s @ 12fps a partir do `video01`
- `caso_critico_cirurgia`: ~25s @ 12fps a partir do `video52`

Idempotente: se o MP4 ja existir, pula. Use `--force` para regenerar.

Uso:
    python scripts/build_demo_videos.py [--force]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_demo_videos")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHOLEC_ROOT = PROJECT_ROOT / "data" / "raw" / "cholecseg8k" / "extracted" / "CholecSeg8k"
EXAMPLES_DIR = PROJECT_ROOT / "data" / "examples"

# Configuracao por caso: video de origem, fps e duracao alvo (segundos).
# A quantidade de frames consumida e int(duracao_s * fps).
CASES: dict[str, dict] = {
    "caso_normal": {
        "video_id": "video01",
        "fps": 12,
        "duracao_s": 15,
    },
    "caso_critico_cirurgia": {
        "video_id": "video52",
        "fps": 12,
        "duracao_s": 25,
    },
}


def _coletar_frames(video_dir: Path, n_frames: int) -> list[Path]:
    """Coleta `n_frames` PNGs `*_endo.png` consecutivos sob `video_dir`.

    Percorre sub-pastas em ordem alfabetica, pegando apenas arquivos que
    nao contenham `_mask` no nome. Quebra assim que o total bate `n_frames`.

    Args:
        video_dir: pasta do tipo `<CHOLEC_ROOT>/<video_id>/`.
        n_frames: quantidade alvo de PNGs a coletar.

    Returns:
        Lista de Paths absolutos, em ordem cronologica.

    Raises:
        FileNotFoundError: se `video_dir` nao existir.
        RuntimeError: se nao houver frames suficientes.
    """
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Pasta de video CholecSeg8k nao existe: {video_dir}")

    subpastas = sorted(p for p in video_dir.iterdir() if p.is_dir())
    coletados: list[Path] = []
    for sub in subpastas:
        frames = sorted(
            p for p in sub.iterdir()
            if p.suffix == ".png" and p.stem.endswith("_endo") and "_mask" not in p.name
        )
        for f in frames:
            coletados.append(f)
            if len(coletados) >= n_frames:
                return coletados[:n_frames]

    if len(coletados) < n_frames:
        raise RuntimeError(
            f"Frames insuficientes em {video_dir}: precisava de {n_frames}, "
            f"encontrei {len(coletados)}."
        )
    return coletados[:n_frames]


def _executar_ffmpeg(staging_dir: Path, destino: Path, fps: int) -> None:
    """Invoca `ffmpeg` para empacotar frames numerados em MP4 H.264.

    Os frames devem estar nomeados `%06d.png` em `staging_dir`. Usa preset
    `medium` e `pix_fmt yuv420p` para compatibilidade ampla com players.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(staging_dir / "%06d.png"),
        "-c:v", "libx264",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(destino),
    ]
    logger.info("Rodando ffmpeg para %s", destino.name)
    subprocess.run(cmd, check=True)


def _gerar_caso(caso: str, config: dict, *, force: bool) -> Path | None:
    """Gera o MP4 do `caso` a partir das configuracoes."""
    destino = EXAMPLES_DIR / caso / "video.mp4"
    if destino.exists() and not force:
        logger.info("[skip] %s ja existe (%d KB)", destino.relative_to(PROJECT_ROOT),
                    destino.stat().st_size // 1024)
        return destino

    n_frames = int(config["duracao_s"] * config["fps"])
    video_id = config["video_id"]
    video_dir = CHOLEC_ROOT / video_id

    try:
        frames = _coletar_frames(video_dir, n_frames)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("[fail] %s: %s", caso, exc)
        return None

    with tempfile.TemporaryDirectory(prefix=f"frames_{caso}_") as tmp:
        staging = Path(tmp)
        for idx, frame in enumerate(frames):
            shutil.copy2(frame, staging / f"{idx:06d}.png")
        try:
            _executar_ffmpeg(staging, destino, config["fps"])
        except subprocess.CalledProcessError as exc:
            logger.error("[fail] %s ffmpeg retornou codigo %d", caso, exc.returncode)
            return None

    tamanho_kb = destino.stat().st_size // 1024
    logger.info("[OK] %s/video.mp4 (%d KB, %d frames)", caso, tamanho_kb, n_frames)
    return destino


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Le os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="Regenera MP4 mesmo se ja existir.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Gera os MP4s dos casos com video."""
    args = _parse_args(argv)
    if shutil.which("ffmpeg") is None:
        logger.error("ffmpeg nao encontrado no PATH. Instale com `brew install ffmpeg`.")
        return 1

    sucesso = 0
    falha = 0
    for caso, config in CASES.items():
        if _gerar_caso(caso, config, force=args.force) is not None:
            sucesso += 1
        else:
            falha += 1

    print(f"Videos gerados: {sucesso} OK, {falha} falha(s).")
    return 0 if falha == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
