"""Gera 3 exemplos pre-carregados em `data/examples/` para a UI Gradio.

Cada cenario combina um audio sintetico existente em `data/synthetic/` com
um trecho curto de video sintetico (com ou sem "instrumental cirurgico" simulado) e um
arquivo `context.txt` com o texto clinico do caso. Um `manifest.json`
agrega os 3 casos para o `app.py` consumir como `gr.Examples`.

Cenarios:
- `caso_normal`: audio normal + video sem deteccoes + contexto neutro
- `caso_moderado`: audio depressao + sem video + contexto com sentimento negativo
- `caso_critico`: audio ansiedade + video com "instrumental cirurgico" simulado
    (retangulo vermelho persistente) + contexto com termo critico ("hemorragia")

Uso:
    python scripts/seed_examples.py
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("seed_examples")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
EXAMPLES_DIR = PROJECT_ROOT / "data" / "examples"


def _gerar_video(destino: Path, *, instrument: bool, duracao_s: float = 4.0, fps: int = 24) -> Path:
    """Gera um video sintetico (.mp4) curto para a UI.

    Quando `instrument=True`, desenha um retangulo vermelho persistente em
    todos os frames simulando uma deteccao de instrumental cirurgico.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    largura, altura = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(destino), fourcc, fps, (largura, altura))
    total = int(duracao_s * fps)
    for i in range(total):
        frame = np.full((altura, largura, 3), 30, dtype=np.uint8)
        x = int((i / total) * (largura - 60))
        cv2.rectangle(frame, (x, 90), (x + 60, 150), (0, 200, 200), -1)
        if instrument:
            cv2.rectangle(frame, (40, 30), (140, 80), (0, 0, 200), -1)
        writer.write(frame)
    writer.release()
    logger.info("Video gerado em %s (%d frames, instrument=%s)", destino, total, instrument)
    return destino


def _copiar(audio_origem: Path, destino: Path) -> Path:
    """Copia um audio sintetico para o diretorio de exemplo (idempotente)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    if not audio_origem.exists():
        raise FileNotFoundError(
            f"Audio sintetico esperado nao existe: {audio_origem}. "
            "Rode `python scripts/gen_synthetic_audio.py` antes."
        )
    shutil.copy2(audio_origem, destino)
    logger.info("Audio copiado para %s", destino)
    return destino


def _escrever_contexto(destino: Path, texto: str) -> Path:
    """Salva o contexto clinico do caso em `context.txt`."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto.strip() + "\n", encoding="utf-8")
    return destino


def _seed_caso_normal() -> dict:
    """Cenario sem anomalia detectada (baseline)."""
    pasta = EXAMPLES_DIR / "caso_normal"
    video = _gerar_video(pasta / "video.mp4", instrument=False)
    audio = _copiar(SYNTHETIC_DIR / "audio_normal.wav", pasta / "audio.wav")
    contexto = (
        "Paciente 28 anos, 24 semanas de gestacao, consulta de rotina. "
        "Sem queixas. Pressao 110x70 mmHg. Auscultacao fetal preservada."
    )
    _escrever_contexto(pasta / "context.txt", contexto)
    return {
        "nome": "Caso normal",
        "descricao": "Consulta de rotina sem queixas.",
        "video_path": str(video.relative_to(PROJECT_ROOT)),
        "audio_path": str(audio.relative_to(PROJECT_ROOT)),
        "context_text": contexto,
        "patient_metadata": {"id": "exemplo-normal", "idade": 28, "semanas": 24},
        "nivel_esperado": "normal",
    }


def _seed_caso_moderado() -> dict:
    """Cenario moderado: queixas afetivas, sem sinal critico no video."""
    pasta = EXAMPLES_DIR / "caso_moderado"
    audio = _copiar(SYNTHETIC_DIR / "audio_depressao.wav", pasta / "audio.wav")
    contexto = (
        "Paciente 32 anos, puerpera 4 semanas. Relata cansaco persistente, "
        "anedonia e dificuldade em vincular-se com o bebe. Sono fragmentado. "
        "Sem queixas fisicas agudas."
    )
    _escrever_contexto(pasta / "context.txt", contexto)
    return {
        "nome": "Caso moderado",
        "descricao": "Suspeita de quadro depressivo puerperal.",
        "video_path": None,
        "audio_path": str(audio.relative_to(PROJECT_ROOT)),
        "context_text": contexto,
        "patient_metadata": {"id": "exemplo-moderado", "idade": 32, "puerperio_semanas": 4},
        "nivel_esperado": "moderate",
    }


def _seed_caso_critico() -> dict:
    """Cenario critico: sangramento simulado + termo critico no contexto."""
    pasta = EXAMPLES_DIR / "caso_critico"
    video = _gerar_video(pasta / "video.mp4", instrument=True)
    audio = _copiar(SYNTHETIC_DIR / "audio_ansiedade.wav", pasta / "audio.wav")
    contexto = (
        "Paciente 35 anos, 36 semanas de gestacao. Chegou ao pronto-socorro "
        "com queixa de hemorragia vaginal intensa nas ultimas duas horas e "
        "dor abdominal severa. Pressao 150x100 mmHg."
    )
    _escrever_contexto(pasta / "context.txt", contexto)
    return {
        "nome": "Caso critico",
        "descricao": "Hemorragia ativa em gestante de termo.",
        "video_path": str(video.relative_to(PROJECT_ROOT)),
        "audio_path": str(audio.relative_to(PROJECT_ROOT)),
        "context_text": contexto,
        "patient_metadata": {"id": "exemplo-critico", "idade": 35, "semanas": 36},
        "nivel_esperado": "critical",
    }


def main() -> int:
    """Cria os 3 exemplos e o manifest na pasta `data/examples/`."""
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    casos = [
        _seed_caso_normal(),
        _seed_caso_moderado(),
        _seed_caso_critico(),
    ]
    manifest_path = EXAMPLES_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps({"casos": casos}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Manifest escrito em %s", manifest_path)
    print(f"Seed concluido: {len(casos)} casos em {EXAMPLES_DIR.relative_to(PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
