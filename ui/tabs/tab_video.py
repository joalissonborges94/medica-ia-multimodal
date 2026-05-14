"""Aba "Vıdeo" da UI Gradio.

Permite upload de um vıdeo, dispara o `VideoPipeline` e mostra os eventos
gerados (deteccoes YOLO, emocao facial, contagem de frames). O pipeline e
fornecido via callback para que o `app.py` possa injetar uma instancia
compartilhada e fazer lazy-load dos modelos.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import gradio as gr

from ui.components import (
    VIDEO_EVENTS_HEADERS,
    build_video_timeline_plot,
    empty_state,
    kpi_grid,
    kpi_tile,
    progress_breadcrumb,
    risk_badge,
    section_title,
    video_events_to_rows,
)

logger = logging.getLogger(__name__)

VideoProcessor = Callable[[Path], "list"]


def render(process_video: VideoProcessor) -> None:
    """Constroi a aba de vıdeo dentro do contexto atual de `gr.Blocks`.

    Args:
        process_video: callable que recebe o path do vıdeo e devolve uma
            `list[VideoEvent]`. Tipicamente `lambda p: orchestrator
            .video_pipeline.process(p)`. Permite injetar mocks em testes.
    """
    gr.HTML(progress_breadcrumb(1))

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            with gr.Group():
                gr.HTML(
                    section_title(
                        "Entrada",
                        "Faca upload de um vıdeo curto (mp4/mov) para extrair "
                        "eventos por frame: deteccoes YOLO, landmarks de pose e "
                        "emocao facial.",
                    )
                )
                video_input = gr.Video(label="Vıdeo de entrada", sources=["upload"])
                analyze_btn = gr.Button("Analisar vıdeo", variant="primary")

        with gr.Column(scale=1):
            with gr.Group():
                gr.HTML(section_title("Resumo"))
                status_html = gr.HTML(
                    value=empty_state(
                        "Nenhum vıdeo analisado ainda.",
                        hint="Faca upload e clique em Analisar para gerar o resumo.",
                    )
                )
                kpis_html = gr.HTML(value="")

    with gr.Group():
        gr.HTML(section_title("Timeline de eventos"))
        timeline_plot = gr.Plot(label="", show_label=False)

    with gr.Group():
        gr.HTML(
            section_title(
                "Eventos por frame",
                "Cada linha e um frame amostrado. Classes e emocoes ficam em branco "
                "quando o frame nao acionou nenhuma deteccao.",
            )
        )
        events_table = gr.Dataframe(
            headers=VIDEO_EVENTS_HEADERS,
            datatype=["str"] * len(VIDEO_EVENTS_HEADERS),
            wrap=True,
            interactive=False,
            value=[],
        )

    with gr.Accordion("JSON bruto (50 primeiros eventos)", open=False):
        raw_json = gr.JSON(value={"events": []})

    def _on_analyze(video_path: str | None):
        if not video_path:
            return (
                empty_state(
                    "Faca upload de um vıdeo antes de analisar.",
                    hint="O arquivo precisa ser mp4 ou mov.",
                ),
                "",
                None,
                [],
                {"events": []},
            )
        try:
            events = process_video(Path(video_path))
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            logger.warning("Falha ao processar vıdeo na UI: %s", exc)
            return (
                empty_state("Falha ao processar vıdeo.", hint=str(exc)),
                "",
                None,
                [],
                {"events": []},
            )

        total_detections = sum(len(e.detections) for e in events)
        emotions = sum(1 for e in events if e.facial_emotion is not None)
        classes_set = {d.class_name for e in events for d in e.detections}
        level = "critical" if "bleeding" in classes_set else "normal"
        status_block = (
            f'<div style="display: flex; align-items: center; gap: 12px;">'
            f"{risk_badge(level)}"
            f'<span style="color: var(--body-text-color); font-size: 13px;">'
            f"Heuristica desta aba (ver aba Multimodal para classificacao completa)."
            f"</span></div>"
        )
        kpis = kpi_grid(
            [
                kpi_tile("Frames", str(len(events)), hint="amostrados"),
                kpi_tile("Deteccoes", str(total_detections), hint="total no video"),
                kpi_tile("Emocoes", str(emotions), hint="frames com face"),
                kpi_tile(
                    "Classes",
                    str(len(classes_set)),
                    hint=", ".join(sorted(classes_set)) if classes_set else "nenhuma",
                ),
            ]
        )
        rows = video_events_to_rows(events)
        payload = {"events": [e.model_dump() for e in events[:50]]}
        timeline = build_video_timeline_plot(events)
        return status_block, kpis, timeline, rows, payload

    analyze_btn.click(
        fn=lambda: gr.update(interactive=False, value="Analisando..."),
        outputs=analyze_btn,
        queue=False,
    ).then(
        fn=_on_analyze,
        inputs=[video_input],
        outputs=[status_html, kpis_html, timeline_plot, events_table, raw_json],
        show_progress="full",
    ).then(
        fn=lambda: gr.update(interactive=True, value="Analisar vıdeo"),
        outputs=analyze_btn,
        queue=False,
    )
