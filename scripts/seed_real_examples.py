"""Orquestra os scripts de seed e regenera o `manifest.json` real.

Executa em sequencia:
    1. `gen_tts_scripts.py`  (WAVs Azure Speech TTS)
    2. `gen_contexts.py`     (TXTs GPT-4.1-mini)

Os MP4s ja vem versionados em `data/examples/<caso>/video.mp4` (commitados
no repo). Em seguida sobrescreve `data/examples/manifest.json` com as 4
entradas dos casos demo (normal, moderado, critico cirurgia, critico
consulta) lendo o conteudo de cada `context.txt`. Casos sem `context.txt`
valido sao pulados do manifest (mas listados no stdout para investigacao).

Limpa pastas obsoletas do esquema antigo (`data/examples/caso_critico/`)
para evitar lixo.

Uso:
    python scripts/seed_real_examples.py [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("seed_real_examples")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "data" / "examples"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Esquema do manifest. Paths sao relativos a raiz do projeto.
CASOS: list[dict] = [
    {
        "key": "caso_normal",
        "nome": "Caso normal (consulta de rotina)",
        "descricao": "Gestante 28 semanas em consulta de rotina, sem queixas.",
        "video_path": "data/examples/caso_normal/video.mp4",
        "audio_path": "data/examples/caso_normal/audio.wav",
        "patient_metadata": {"id": "exemplo-normal", "idade": 28, "semanas": 28},
        "nivel_esperado": "normal",
    },
    {
        "key": "caso_moderado",
        "nome": "Caso moderado (depressao pos-parto)",
        "descricao": "Puerpera 4 semanas com anedonia e queixa afetiva.",
        "video_path": None,
        "audio_path": "data/examples/caso_moderado/audio.wav",
        "patient_metadata": {"id": "exemplo-moderado", "idade": 32, "puerperio_semanas": 4},
        "nivel_esperado": "moderate",
    },
    {
        "key": "caso_critico_cirurgia",
        "nome": "Caso critico (cirurgia em andamento)",
        "descricao": "Laparoscopia com ansiedade pre-procedimento.",
        "video_path": "data/examples/caso_critico_cirurgia/video.mp4",
        "audio_path": "data/examples/caso_critico_cirurgia/audio.wav",
        "patient_metadata": {"id": "exemplo-critico-cirurgia", "idade": 38},
        "nivel_esperado": "critical",
    },
    {
        "key": "caso_critico_consulta",
        "nome": "Caso critico (hemorragia em gestante de termo)",
        "descricao": "Gestante 36 semanas com sangramento intenso e PA elevada.",
        "video_path": None,
        "audio_path": "data/examples/caso_critico_consulta/audio.wav",
        "patient_metadata": {"id": "exemplo-critico-consulta", "idade": 36, "semanas": 36},
        "nivel_esperado": "critical",
    },
]

# Pastas do esquema antigo que devem ser removidas para nao confundir o app.
PASTAS_OBSOLETAS = ["caso_critico"]


def _rodar(script: str, extra: list[str]) -> int:
    """Executa `python scripts/<script>` como subprocess e retorna o exit code."""
    cmd = [sys.executable, str(SCRIPTS_DIR / script), *extra]
    logger.info("--> %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def _ler_contexto(caso_key: str) -> str | None:
    """Le `data/examples/<caso>/context.txt` se existir e nao for vazio."""
    path = EXAMPLES_DIR / caso_key / "context.txt"
    if not path.exists():
        return None
    texto = path.read_text(encoding="utf-8").strip()
    return texto or None


def _limpar_obsoletas() -> None:
    """Remove pastas do esquema antigo que nao fazem parte do novo manifest."""
    for nome in PASTAS_OBSOLETAS:
        pasta = EXAMPLES_DIR / nome
        if pasta.exists():
            shutil.rmtree(pasta)
            logger.info("[clean] removida pasta obsoleta %s", pasta.relative_to(PROJECT_ROOT))


def _gerar_manifest() -> tuple[int, int]:
    """Monta `manifest.json` com os casos que tem `context.txt` valido."""
    casos_saida: list[dict] = []
    pulados = 0
    for caso in CASOS:
        key = caso["key"]
        texto = _ler_contexto(key)
        if texto is None:
            logger.warning("[skip-manifest] %s sem context.txt valido", key)
            pulados += 1
            continue
        entrada = {
            "nome": caso["nome"],
            "descricao": caso["descricao"],
            "video_path": caso["video_path"],
            "audio_path": caso["audio_path"],
            "context_text": texto,
            "patient_metadata": caso["patient_metadata"],
            "nivel_esperado": caso["nivel_esperado"],
        }
        casos_saida.append(entrada)

    manifest_path = EXAMPLES_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps({"casos": casos_saida}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Manifest com %d caso(s) escrito em %s",
                len(casos_saida), manifest_path.relative_to(PROJECT_ROOT))
    return len(casos_saida), pulados


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Le os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="Forca regerar todos os artefatos (passa --force aos sub-scripts).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Roda os 3 scripts, limpa obsoletos e regenera o manifest."""
    args = _parse_args(argv)
    extra = ["--force"] if args.force else []

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    rc_tts = _rodar("gen_tts_scripts.py", extra)
    rc_ctx = _rodar("gen_contexts.py", extra)

    _limpar_obsoletas()
    incluidos, pulados = _gerar_manifest()

    print(
        f"Seed real concluido: {incluidos} caso(s) no manifest, {pulados} pulado(s). "
        f"Exit codes: tts={rc_tts}, contextos={rc_ctx}."
    )
    if incluidos == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
