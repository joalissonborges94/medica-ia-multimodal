# medica-ia-multimodal

Sistema de monitoramento multimodal aplicado à saúde da mulher. Tech Challenge Fase 4 da pós-graduação Tech IADT.

Combina análise de vídeo (YOLOv8 + MediaPipe + FER), áudio (faster-whisper + librosa + wav2vec2), RAG sobre diretrizes clínicas brasileiras (Chroma + bge-m3) e LLM (Azure OpenAI) para detectar anomalias e gerar relatórios clínicos com alertas categorizados.

## O Que Faz

Recebe vídeo, áudio e/ou contexto textual sobre uma paciente. Processa cada modalidade em pipelines especializados, funde os resultados, consulta diretrizes clínicas via RAG e devolve:

- **Nível de risco** categorizado (normal, moderado, crítico)
- **Relatório clínico** em markdown com achados e recomendações
- **Alerta estruturado** quando moderado ou crítico
- **Auditoria** persistente em SQLite

UI Gradio com 4 abas: **Vídeo**, **Áudio**, **Multimodal** e **Auditoria**. Sem credenciais Azure, o sistema usa fallback determinístico e ainda demonstra o fluxo completo.

## Como Rodar

Pré-requisitos: Python 3.12, ~6 GB livres em disco (modelos + dataset).

```bash
# 1. Clonar e entrar no diretório
git clone <repo-url> medica-ia-multimodal && cd medica-ia-multimodal

# 2. Criar venv e instalar dependências
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configurar variáveis de ambiente (opcional; o app roda offline com fallback)
cp .env.example .env
# Editar .env preenchendo as chaves Azure que você tiver

# 4. Setup completo em um comando: modelos + dataset + PDFs + índice RAG + exemplos UI
python scripts/warmup.py

# 5. Subir a UI Gradio
python app.py
# Abrir http://127.0.0.1:7860
```

O `warmup.py` é idempotente e aceita filtros: `--models`, `--datasets`, `--pdfs`, `--examples`, `--skip-rag`. Detalhes em `python scripts/warmup.py --help`.

## Integrações Azure

Quando as chaves estão preenchidas no `.env`, o sistema usa os serviços gerenciados abaixo; sem elas, cada pilar cai num fallback local equivalente.

| Serviço | Função | Onde é usado | Fallback offline |
|---|---|---|---|
| **Azure OpenAI** (GPT-4.1-mini, AI Foundry) | Gera o relatório clínico final em markdown | `src/llm/azure_openai.py`, `src/report.py` | Relatório determinístico em markdown construído a partir das triggers |
| **Azure Speech** | Transcrição de áudio + TTS para gerar voz PT-BR | `src/audio/transcriber.py` (toggle `USE_CLOUD_TRANSCRIPTION`) | `faster-whisper` local |
| **Azure Language** | Análise de sentimento + key phrases na transcrição | `src/audio/azure_language.py` | Pular esse pilar (não há substituto local equivalente) |
| **Azure Face** | Emoção facial em vídeo (adiado por RAI policy) | n/a | `FER` local (Py 3.12) |

A aba **Configurações** da UI mostra em tempo real quais serviços estão ativos (cloud) ou em fallback (local).

## Docker

```bash
docker compose up
```

## Testes

```bash
pytest -m smoke        # rápidos, com mocks
pytest -m integration  # end-to-end com pipelines mockados
ruff check src/ ui/ tests/ scripts/ app.py
```

## Documentação

- [docs/overview.md](docs/overview.md): visão geral, escopo, métricas de sucesso
- [docs/arquitetura/arquitetura.md](docs/arquitetura/arquitetura.md): módulos, fluxo, interfaces
- [docs/arquitetura/decisoes_tecnicas.md](docs/arquitetura/decisoes_tecnicas.md): ADRs com justificativa
- [docs/arquitetura/modelos_e_datasets.md](docs/arquitetura/modelos_e_datasets.md): modelos e fontes de dados
- [docs/arquitetura/padroes_codigo.md](docs/arquitetura/padroes_codigo.md): convenções de código

## Equipe

Joalisson Borges, Luis Gustavo Santini, Marina Souza Lucas, Diego Santos.

## Licença

MIT. Ver [LICENSE](LICENSE).
