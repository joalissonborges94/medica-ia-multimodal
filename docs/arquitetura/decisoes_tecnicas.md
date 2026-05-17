# Decisões Técnicas

Registro de decisões importantes no formato ADR leve (Architecture Decision Record). Cada decisão tem contexto, opções consideradas, decisão tomada e consequências.

Atualizar conforme novas decisões surgem durante a implementação.

---

## ADR-001: Não reaproveitar código da Fase 3

**Contexto:** A Fase 3 entregou um assistente médico geral com Qwen3.5-4B fine-tunado, LangGraph + ReAct, MIMIC-IV. A Fase 4 muda o domínio para saúde da mulher e adiciona análise multimodal.

**Opções:**

A. Estender o pipeline LangGraph da Fase 3 adicionando módulos multimodais
B. Construir solução nova focada em análise multimodal
C. Híbrida: usar LLM da Fase 3 e construir o resto novo

**Decisão:** Opção B.

**Justificativa:** o núcleo técnico da Fase 4 é diferente da Fase 3. Reaproveitar adiciona mais ônus (Ollama no Colab, fine-tuning genérico, LangGraph com 6 nós que não fazem mais sentido) do que ganho.

**Consequências:** notebook menor, arquitetura limpa, onboarding rápido. Perde narrativa de continuidade que será mencionada no relatório técnico mas não no código.

---

## ADR-002: UI em Gradio Blocks ao invés de React + FastAPI

**Contexto:** A entrega exige um vídeo demo de 15 minutos mostrando o sistema funcionando. UI ajuda a apresentar.

**Opções:**

A. Gradio (Interface ou Blocks)
B. Streamlit
C. React + FastAPI separados

**Decisão:** Gradio Blocks.

**Justificativa:** trade-off tempo vs qualidade. Gradio Blocks permite layout custom, tema próprio, componentes ML especializados (Video, Audio, Plot) e roda em 1-2 dias. React + FastAPI levaria 5-10 dias para resultado equivalente.

**Consequências:** UI fica menos "produto real" mas suficiente para a demo. Backend e frontend no mesmo processo, deploy unificado.

---

## ADR-003: Hospedagem em Hugging Face Spaces

**Contexto:** precisa de URL público para a demo e para incluir no README.

**Opções:**

A. Hugging Face Spaces (free)
B. Azure Container Apps (pago)
C. Cloud Run (pago)
D. Apenas local com Docker

**Decisão:** Hugging Face Spaces.

**Justificativa:** zero custo, deploy via git push, suporta Gradio nativo, link público estável. Hibernação ao ocioso é aceitável para demo.

**Consequências:** uso limitado a 16GB RAM e 2 vCPU. Não tem GPU mas a inferência foi planejada para CPU.

---

## ADR-004: Provider de cloud é Azure (não AWS)

**Contexto:** o enunciado cita Azure Cognitive Services, mas a grade da fase ensina AWS Textract + Comprehend.

**Opções:**

A. Azure puro
B. AWS puro
C. Híbrido Azure + AWS

**Decisão:** Azure puro.

**Justificativa:** o vídeo demo da entrega lista explicitamente "Integração dos serviços Azure". Tratar como obrigatório é mais seguro do que apostar na interpretação literal do enunciado.

**Consequências:** AWS Textract e Comprehend Medical ficam fora do escopo, não cobrindo a parte da grade que ensina esses serviços. O relatório técnico justifica a escolha.

---

## ADR-005: Provider de LLM é Azure OpenAI

**Contexto:** projeto precisa de LLM para sumarização clínica e geração de relatórios.

**Opções:**

A. OpenAI direto
B. Azure OpenAI
C. Modelos open source local

**Decisão:** Azure OpenAI.

**Justificativa:** mantém tudo no ecossistema Azure, reforça o requisito de integração Azure do desafio. Mesmos modelos GPT-4.1-mini e Whisper, com SLA empresarial.

**Consequências:** precisa provisionar deployment de modelo no Azure OpenAI (via AI Foundry, caminho de provisionamento adotado no 2026-05-14, sem espera de aprovação manual). Custo controlado por limites no painel.

---

## ADR-006: YOLO base pré-treinado primeiro, custom depois

**Contexto:** treinar YOLO custom desde o início bloqueia o desenvolvimento das outras frentes (UI, fusão, RAG).

**Opções:**

A. Treinar YOLO custom antes de tudo o mais
B. Usar YOLOv8 base do ultralytics como stub
C. Usar YOLO pré-treinado clínico do Roboflow Universe

**Decisão:** Opção C primeiro, depois treino custom em etapa posterior.

**Justificativa:** padrão "stub the model" destrava o desenvolvimento. Detector é model-agnostic, troca de pesos não afeta arquitetura. Roboflow Universe oferece modelos clínicos prontos.

**Consequências:** o pipeline inicial começa rápido; a etapa de treino custom entrega o modelo final.

---

## ADR-007: YOLO target é sangramento anômalo

> **SUPERSEDED por [ADR-012](#adr-012-yolo-custom-muda-alvo-para-instrumentos-cirúrgicos-cholecseg8k).** Registro mantido por histórico de decisão.

**Contexto:** o enunciado lista 4 alvos possíveis para o YOLOv8 customizado.

**Opções:**

A. Instrumentos cirúrgicos ginecológicos
B. Áreas críticas (útero, ovários, mamas)
C. Sangramento anômalo durante procedimentos
D. Objetos suspeitos de automutilação

**Decisão:** Opção C, sangramento anômalo.

**Justificativa:** alinhamento direto com saúde materna, maior chance de obter dado rotulado público, relevância clínica imediata, baixo risco ético comparado a opção D.

**Consequências:** dataset precisa ser obtido via Roboflow ou anotação manual em pequena escala.

---

## ADR-008: Vector store Chroma para RAG

**Contexto:** RAG precisa de vector store funcional para diretrizes clínicas.

**Opções:**

A. Chroma persistente
B. FAISS
C. pgvector
D. Pinecone

**Decisão:** Chroma persistente.

**Justificativa:** zero setup, persiste em arquivo, integra direto com LangChain, suporta filtros por metadata. Suficiente para algumas dezenas de PDFs.

**Consequências:** se o volume crescer muito, migrar para FAISS ou pgvector. Não é o caso aqui.

---

## ADR-009: Idioma do código em EN, comentários e docs em PT-BR

**Contexto:** projeto tem nomenclatura técnica padrão e domínio clínico brasileiro.

**Opções:**

A. Tudo em PT-BR
B. Tudo em EN
C. Híbrido: nomes em EN, docs e comentários em PT-BR

**Decisão:** Opção C.

**Justificativa:** nomes em EN seguem padrão da indústria e libs externas. Comentários e docs em PT-BR são mais naturais para o domínio clínico brasileiro e para o avaliador.

**Consequências:** algumas mensagens de log e variáveis terão mistura. Manter consistência interna por módulo.

---

## ADR-010: Audit log em SQLite

**Contexto:** sistema precisa registrar decisões e ações para auditoria.

**Opções:**

A. SQLite local
B. Postgres em container
C. Logs em arquivos JSON

**Decisão:** SQLite local.

**Justificativa:** zero setup, query SQL, persiste, tamanho irrisório. Para auditoria de eventos de sessão é perfeito.

**Consequências:** não escala para múltiplos usuários simultâneos, mas o caso de uso atual é demonstração.

---

## ADR-011: Stub `yolov8n.pt` inicial; Roboflow Universe avaliado para etapa de treino custom

**Contexto:** ADR-006 escolheu Roboflow Universe como fonte do modelo pré-treinado para destravar o pipeline inicial. Na prática, garimpar um modelo público de "bleeding" no Universe com pesos baixáveis, license permissiva e mAP confiável tem custo de tempo alto, e o Universe ainda bloqueia automação externa. Isso bloqueia o pipeline inicial inteiro por algo que é desnecessário no curto prazo: o detector é model-agnostic e o treino custom está planejado para etapa posterior de qualquer jeito.

**Opções:**

A. Continuar buscando modelo Roboflow ideal antes de destravar o pipeline
B. Stub com `yolov8n.pt` base (Ultralytics) imediatamente, trocar pelos pesos custom na etapa de treino
C. Stub com `yolov8n.pt` agora E pesos Roboflow assim que aparecerem como melhoria opcional

**Decisão:** Opção B.

**Justificativa:** o treino custom está planejado para etapa posterior (com dataset rotulado vindo de Roboflow Universe ou outras fontes). Não há ganho real em ter modelo "intermediário Roboflow" entre `yolov8n` base e o custom final. Roboflow Universe segue como **fonte de dataset** (não de modelo) na etapa de treino.

**Consequências:**
- O pipeline inicial desbloqueia imediatamente, sem dependência externa.
- O detector aceita qualquer `.pt` via `YOLO_WEIGHTS_PATH` (env var), mantendo a model-agnosticidade prometida no ADR-006.
- ADR-006 letra C continua válido para a etapa de treino custom: avaliar Roboflow Universe como dataset; alternativas (Kaggle, HuggingFace, PhysioNet) ficam abertas.
- Stub atual detecta classes COCO (não `bleeding`). O pipeline roda end-to-end mesmo assim; a relevância clínica vem do treino custom.

---

## ADR-012: YOLO custom muda alvo para instrumentos cirúrgicos (CholecSeg8k)

**Supersede:** ADR-007.

**Contexto:** ADR-007 escolheu "sangramento anômalo" como alvo do YOLO custom. Ao executar o provisionamento de dataset, uma rodada de pesquisa empírica em fontes públicas (Roboflow Universe, Kaggle, Hugging Face, PhysioNet, repositórios académicos) revelou que nenhuma das opções viáveis sustenta o alvo "sangramento" no domínio ginecológico:

- **WCEBleedGen (Kaggle/Wireless Capsule Endoscopy):** sangramento sim, mas em gastroenterologia (cápsula endoscópica). Domínio não-ginecológico, framing forçado.
- **BUSI (Breast Ultrasound Images):** ultrassom mamário diagnóstico, sem sangramento. Substituir "sangramento" por "lesão mamária" seria mudança implícita, sem aderência ao enunciado.
- **Dresden Surgical Anatomy Dataset:** alvo cirúrgico ginecológico real, mas 19 GB com licença restritiva e pipeline de download inviável para reprodutibilidade em Colab/HF Spaces.
- **CholecSeg8k (Hong et al., arXiv 2012.12463, CC BY-NC-SA 4.0):** 8080 frames anotados de colecistectomia laparoscópica, 3.1 GB, disponível anônimo via Hugging Face Datasets. Contém **Grasper** e **L-hook Electrocautery** anotados, exatamente os instrumentos usados em cirurgia ginecológica laparoscópica.

**Opções:**

A. Manter "sangramento" mesmo sem dataset ginecológico real (forçar com WCEBleedGen)
B. Pivotar para "lesão mamária" usando BUSI (US diagnóstico)
C. Pivotar para "instrumentos cirúrgicos" usando Dresden (19 GB inviável)
D. Pivotar para "instrumentos cirúrgicos" usando CholecSeg8k (3.1 GB, HF, classes adequadas)

**Decisão:** Opção D. YOLO custom passa a detectar **instrumentos cirúrgicos** (classes: `Grasper`, `L-hook Electrocautery`) treinado em CholecSeg8k.

**Justificativa:**
- Aderência LITERAL ao alvo 1 do enunciado ("Instrumentos cirúrgicos ginecológicos"). Foi a única opção do enunciado que sustenta dado público real.
- Reprodutibilidade: 3.1 GB cabe no fluxo Colab + Hugging Face, sem login proprietário ou EULA bloqueante.
- Técnica laparoscópica idêntica entre colecistectomia (treino) e cirurgia ginecológica (aplicação): mesmo trocarte, mesma pinça Grasper, mesmo L-hook eletrocautério. Transferência de domínio justificada clinicamente.
- Licença CC BY-NC-SA 4.0 compatível com uso académico/demonstração.

**Consequências:**
- Dataset baixado em `data/raw/cholecseg8k/`, convertido para formato YOLO em `data/processed/cholecseg8k_yolo/` (8080 frames, 2 classes, splits 5656/1616/808 train/val/test).
- Refactor em código: `LESION_MAMA_*` → `SURGICAL_INSTRUMENT_*` em `src/anomaly/rules.py`, `src/anomaly/statistical.py` e respectivos testes.
- **Semântica nova:** detecção de instrumento é estado NORMAL de cirurgia laparoscópica. Trigger correspondente eleva nível para `moderate` apenas (sinaliza presença de procedimento invasivo em curso); nunca para `critical` isoladamente. Combinação com outros triggers (vocais, textuais) pode escalar.
- ADR-007 superseded. Histórico de pivots preservado neste registro: WCEBleedGen → BUSI → Dresden → CholecSeg8k.

---

## ADR-013: Estratégia de áudio (TTS Azure scriptado + validação CORAA-SER)

**Contexto:** o pipeline de áudio precisa de exemplos clínicos em PT-BR para demonstrar emoção vocal, transcrição e classificação de afeto. Não existe corpus público brasileiro de consultas ginecológicas/obstétricas reais com rótulos de emoção (pacientes reais implicam LGPD e comité de ética, inviável no escopo do challenge). Corpora estrangeiros (RAVDESS, IEMOCAP) são em inglês e atuados, e o pipeline Whisper + wav2vec2 precisa de PT-BR para ser convincente.

**Opções:**

A. Apenas TTS Azure scriptado (controlado mas artificial)
B. Apenas CORAA-SER (PT-BR espontâneo mas sem cenários clínicos específicos)
C. Híbrida: TTS Azure como "rótulo gold" da demo + CORAA-SER como validação de generalização

**Decisão:** Opção C.

**Justificativa:**
- TTS Azure (vozes Francisca, Antonio, Brenda, etc., com estilos `sad`, `empathetic`, `terrified`) produz áudios PT-BR controlados, com rótulo de emoção determinístico e cenário clínico scripted. Serve como gold standard para a demo.
- CORAA-SER (`alefiury/CORAA-SER` no Hugging Face) fornece fala espontânea PT-BR rotulada em emoção. Usado para validar que o classificador wav2vec2 não overfita ao timbre sintético do TTS.
- Estratégia híbrida mantém demo previsível (TTS) e responsabilidade técnica (validação em fala real).

**Consequências:**
- O projeto gera áudios TTS Azure em `data/synthetic/audio/` cobrindo cenários: normal, ansiedade, depressão, sofrimento agudo.
- `tests/integration/test_audio_validation.py` (a criar) roda subset de CORAA-SER e reporta acurácia agregada do classificador.
- TTS Azure exige `AZURE_SPEECH_KEY` (já provisionado). CORAA-SER baixado on-demand pelo `datasets` da HF.

---

## ADR-014: 3ª funcionalidade (sinais vitais) marcada como Adiado

**Contexto:** o enunciado pede no mínimo 2 funcionalidades dentre as 4 listadas. O projeto já cobre 3 funcionalidades atuais: análise de vídeo, processamento de áudio e integração com Azure Cognitive Services. A 4ª opção do enunciado é "Monitoramento de sinais vitais" (integração com dados de cardiotocografia, pressão arterial, ECG, etc.).

**Opções:**

A. Implementar sinais vitais como 4ª modalidade (cardiotocografia ou PA via CSV/JSON sintético)
B. Adiar formalmente, marcar como `⏸️ Adiado` no roadmap

**Decisão:** Opção B.

**Justificativa:**
- Estimativa de esforco adicional significativo (modelagem de serie temporal + nova aba + novos triggers + integracao no orquestrador) para ganho marginal frente ao requisito de >=2 funcionalidades ja cumprido.
- Quebra coerência multimodal: vídeo + áudio + texto formam uma análise unificada (humana, qualitativa); série temporal de PA/CTG é outro paradigma (numérico, monitoramento contínuo) e exigiria UX própria.
- Aderência ao padrão "stub-first + fallback gracioso": registrar como adiado é melhor do que entregar versão parcial.

**Consequências:**
- Roadmap exibe `⏸️ Adiado` para sinais vitais; mencionado em "Próximos Passos para Evoluir" em `arquitetura.md`.
- Relatório técnico e vídeo demo explicitam que 3 funcionalidades já superam o mínimo do enunciado.
- Caso outro grupo do trio ou avaliador peça, a 4ª funcionalidade fica pré-arquitetada para implementação futura sem refactor.

---

## ADR-015: Substituicao de MediaPipe Pose por YOLOv8 Pose (multi-person)

**Contexto:** `src/video/pose.py` usava `mediapipe.solutions.pose` para estimar landmarks corporais em cenas de consulta. Dois problemas identificados em producao:

1. MediaPipe Pose detecta apenas **uma** pessoa por frame. Em consultas medico + paciente, o detector frequentemente capturava o medico (geralmente mais frontal/proximo da camera) em vez da paciente.
2. No PyPI do Python 3.14 o pacote `mediapipe` vem sem o submodulo `solutions.pose`. A dependencia bloqueava o pipeline em ambientes 3.14 com erro de `AttributeError`.

**Opcoes:**

A. Manter MediaPipe Pose com pin de versao e restringir a Py 3.12
B. Substituir por YOLOv8 Pose (Ultralytics), que ja e dependencia do projeto
C. Implementar pose via Azure Vision (custo adicional por chamada)

**Decisao:** Opcao B.

**Justificativa:**
- YOLOv8 Pose detecta multiplas pessoas simultaneamente. A pessoa principal (maior bbox) e selecionada como paciente, heuristica mais robusta que single-person.
- 17 keypoints COCO cobrem todos os landmarks que `classify_posture` usa (nariz, ombros, quadris). Sem perda funcional em relacao aos 33 do MediaPipe.
- Ultralytics ja estava instalado como dependencia do detector YOLO custom. Zero dependencia nova.
- Funciona uniformemente em Py 3.12 e 3.14 sem condicional de versao.
- Modelo `yolov8n-pose.pt` (~6 MB) e baixado automaticamente na primeira chamada, alinhado com o padrao de lazy-load ja adotado no projeto.

**Consequencias:**
- `classify_posture` continua inalterada: aceita lista de `PoseLandmark` com nomes COCO ou MediaPipe indistintamente.
- Coordenadas normalizadas (0-1) mantidas, compativel com o resto do pipeline.
- Reduz o numero de dependencias do projeto (mediapipe removido do `requirements.txt`).
- `scene_classifier.py` ainda usa `mediapipe.solutions.face_detection` para detectar faces na heuristica de tipo de cena. Esse uso e mais leve (apenas face, sem pose) e permanece com lazy import e fallback gracioso quando mediapipe nao esta disponivel.

---

## Como Adicionar Nova ADR

1. Próximo número sequencial (ADR-011, etc.)
2. Título descritivo da decisão
3. Contexto explicando o problema
4. Opções consideradas (mínimo 2)
5. Decisão tomada com justificativa
6. Consequências (positivas e negativas)
7. Linkar de outros docs quando relevante
