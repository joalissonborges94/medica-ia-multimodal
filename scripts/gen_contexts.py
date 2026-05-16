"""Gera contextos clinicos PT-BR para os 4 casos demo via GPT-4.1-mini.

Usa `src.llm.azure_openai.AzureOpenAIClient` (endpoint Foundry v1).
Cada contexto sai com 5 a 8 linhas curtas cobrindo idade, semanas de
gestacao ou puerperio, queixa principal, sinais vitais e historico breve.

O nivel esperado (normal, moderate, critical) e injetado no prompt para
manter a coerencia clinica. O modelo recebe instrucao explicita para NAO
inventar nomes proprios nem datas absolutas.

Salva em `data/examples/<caso>/context.txt`. Idempotente: pula casos com
arquivo nao vazio; use `--force` para regerar.

Se a chamada ao LLM falhar (chave invalida, timeout, RAI block), o caso
afetado e pulado e os demais continuam.

Uso:
    python scripts/gen_contexts.py [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.azure_openai import AzureOpenAIClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gen_contexts")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "data" / "examples"

# Cada caso define o perfil clinico que o modelo deve respeitar.
CASES: dict[str, dict[str, str]] = {
    "caso_normal": {
        "nivel": "normal",
        "perfil": (
            "Gestante de 28 anos, 28 semanas de gestacao, consulta de rotina "
            "de pre-natal. Sem queixas, pressao normal, exames anteriores OK."
        ),
    },
    "caso_moderado": {
        "nivel": "moderate",
        "perfil": (
            "Puerpera de 32 anos, 4 semanas pos-parto. Queixa de anedonia, "
            "choro frequente, dificuldade de vinculo com o bebe, sono "
            "fragmentado. Suspeita de depressao pos-parto."
        ),
    },
    "caso_critico_cirurgia": {
        "nivel": "critical",
        "perfil": (
            "Paciente de 38 anos em cirurgia laparoscopica em andamento "
            "(colecistectomia). Equipe relata ansiedade pre-procedimento alta. "
            "Pressao limitrofe, frequencia cardiaca elevada."
        ),
    },
    "caso_critico_consulta": {
        "nivel": "critical",
        "perfil": (
            "Gestante de 36 anos, 36 semanas de gestacao. Chegou ao "
            "pronto-socorro com hemorragia vaginal intensa, dor abdominal "
            "severa, PA 150x100 mmHg. Suspeita de descolamento prematuro."
        ),
    },
}

SYSTEM_PROMPT = (
    "Voce e um redator clinico que escreve resumos de prontuario obstetrico "
    "em portugues do Brasil para apresentacoes academicas. Use linguagem "
    "tecnica concisa. NUNCA invente nomes proprios, datas absolutas, hospitais "
    "ou identificadores. Nao use emoji. Nao use travessao (em-dash ou en-dash). "
    "Use ponto, virgula, dois-pontos ou ponto-e-virgula."
)

USER_TEMPLATE = (
    "Escreva um contexto clinico realista para o caso abaixo, em 5 a 8 linhas "
    "curtas (uma frase por linha). Cubra: idade, semanas de gestacao ou "
    "puerperio, queixa principal, sinais vitais relevantes, historico breve. "
    "Mantenha tom compativel com nivel de risco '{nivel}'. Nao inclua "
    "cabecalhos, listas ou markdown: apenas texto corrido em linhas separadas.\n\n"
    "Perfil base: {perfil}"
)


def _gerar_contexto(client: AzureOpenAIClient, caso: str, config: dict[str, str]) -> str | None:
    """Chama o LLM e retorna o texto, ou None em falha."""
    prompt = USER_TEMPLATE.format(nivel=config["nivel"], perfil=config["perfil"])
    resposta = client.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=400,
    )
    if resposta is None or not resposta.strip():
        logger.error("[fail] %s: LLM retornou vazio", caso)
        return None
    return resposta.strip()


def _salvar(caso: str, texto: str) -> Path:
    """Persiste o contexto em `data/examples/<caso>/context.txt`."""
    destino = EXAMPLES_DIR / caso / "context.txt"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto + "\n", encoding="utf-8")
    return destino


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Le os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="Regenera context.txt mesmo se ja existir.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Gera os 4 contextos clinicos."""
    args = _parse_args(argv)
    client = AzureOpenAIClient()
    if not client.is_configured:
        logger.error(
            "Azure OpenAI nao configurado. Defina AZURE_OPENAI_KEY, "
            "AZURE_OPENAI_ENDPOINT e AZURE_OPENAI_DEPLOYMENT em .env."
        )
        return 1

    sucesso = 0
    falha = 0
    for caso, config in CASES.items():
        destino = EXAMPLES_DIR / caso / "context.txt"
        if destino.exists() and destino.stat().st_size > 0 and not args.force:
            logger.info("[skip] %s ja existe (%d bytes)",
                        destino.relative_to(PROJECT_ROOT), destino.stat().st_size)
            sucesso += 1
            continue
        try:
            texto = _gerar_contexto(client, caso, config)
        except Exception as exc:  # noqa: BLE001 - SDK lanca varias subclasses
            logger.error("[fail] %s: excecao %s", caso, exc)
            texto = None
        if texto is None:
            falha += 1
            continue
        _salvar(caso, texto)
        tamanho_bytes = (EXAMPLES_DIR / caso / "context.txt").stat().st_size
        logger.info("[OK] %s/context.txt (%d bytes)", caso, tamanho_bytes)
        sucesso += 1

    print(f"Contextos gerados: {sucesso} OK, {falha} falha(s).")
    return 0 if falha == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
