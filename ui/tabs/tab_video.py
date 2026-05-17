"""Aba "Vıdeo" da UI Gradio.

Permite upload de um vıdeo, dispara o `VideoPipeline` e mostra:

- Resumo (badge de risco + tipo de cena detectado + fonte do classifier de emocao)
- KPIs (frames, deteccoes, emocoes, classes, postura)
- Miniatura do frame com bboxes desenhadas (evidencia visual do que o YOLO viu)
- Timeline de eventos
- Eventos por frame em tabela
- JSON bruto colapsado

Recebe o `VideoPipeline` direto (em vez de so um callable) pra acessar:
- `pipeline.process(path)` para os eventos
- `pipeline.last_scene_type` para o badge de cena
- `pipeline.emotion_classifier` para identificar fonte (Azure ou FER local)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr

from ui.components import (
    VIDEO_WINDOWS_HEADERS,
    build_detection_thumbnails,
    build_pose_thumbnails,
    build_video_timeline_plot,
    emotion_source_label,
    empty_state,
    kpi_grid,
    kpi_tile,
    progress_breadcrumb,
    risk_badge,
    scene_type_badge,
    section_title,
    video_events_to_windowed_rows,
)
from ui.limits import validate_video

if TYPE_CHECKING:
    from src.video.pipeline import VideoPipeline

logger = logging.getLogger(__name__)


def render(video_pipeline: VideoPipeline) -> None:
    """Constroi a aba de vıdeo dentro do contexto atual de `gr.Blocks`.

    Args:
        video_pipeline: instancia compartilhada de `VideoPipeline` (do
            orquestrador). Permite acesso a metadata pos-execucao
            (last_scene_type, emotion_classifier) para enriquecer o Resumo.
    """
    gr.HTML(progress_breadcrumb(1))

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            with gr.Group():
                gr.HTML(
                    section_title(
                        "Entrada",
                        "Faca upload de um vıdeo curto (mp4/mov, recomendado <60s) "
                        "para extrair eventos por frame: deteccoes YOLO, landmarks "
                        "de pose e estado emocional via linguagem corporal.",
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

    detection_section = gr.Group(visible=False)
    with detection_section:
        gr.HTML(
            section_title(
                "Frames com deteccoes",
                "Top 4 frames com mais deteccoes, ordenados temporalmente. "
                "Bboxes coloridas por classe (grasper=verde, l_hook=magenta, "
                "blood=vermelho).",
            )
        )
        detection_thumb = gr.Gallery(
            label="",
            show_label=False,
            columns=4,
            rows=1,
            height="auto",  # ajusta natural à proporção dos frames (sem scroll)
            object_fit="contain",
            allow_preview=True,
            show_share_button=False,
            show_download_button=False,
            elem_classes="compact-gallery",
        )

    pose_section = gr.Group(visible=False)
    with pose_section:
        gr.HTML(
            section_title(
                "Frames com pose",
                "Top 4 frames com mais keypoints visiveis, ordenados temporalmente. "
                "Bbox azul = pessoa principal; skeleton verde = esqueleto COCO; "
                "pontos amarelos = keypoints (visibility >= 0.3).",
            )
        )
        pose_thumb = gr.Gallery(
            label="",
            show_label=False,
            columns=4,
            rows=1,
            height="auto",
            object_fit="cover",
            allow_preview=True,
            show_share_button=False,
            show_download_button=False,
            elem_classes="compact-gallery",
        )

    with gr.Group():
        gr.HTML(section_title("Timeline de eventos"))
        timeline_plot = gr.Plot(label="", show_label=False)

    with gr.Group():
        gr.HTML(
            section_title(
                "Eventos por janela",
                "Resumo agregado em janelas de 5s: deteccoes acumuladas, classes "
                "com confianca media, emocao e postura predominantes. JSON bruto "
                "abaixo tem os eventos cru por frame.",
            )
        )
        events_table = gr.Dataframe(
            headers=VIDEO_WINDOWS_HEADERS,
            datatype=["str"] * len(VIDEO_WINDOWS_HEADERS),
            wrap=True,
            interactive=False,
            value=[],
            elem_classes="compact-table",  # colapsa espaco "pra crescer" do Gradio
        )

    with gr.Accordion("JSON bruto (50 primeiros eventos)", open=False):
        raw_json = gr.JSON(value={"events": []})

    def _on_analyze(
        video_path: str | None,
        progress=gr.Progress(),  # noqa: B008  # pattern idiomatico do Gradio
    ):
        empty_result = (
            empty_state(
                "Faca upload de um vıdeo antes de analisar.",
                hint="O arquivo precisa ser mp4 ou mov.",
            ),
            "",
            None,
            None,
            None,
            None,
            [],
            {"events": []},
        )
        if not video_path:
            return empty_result

        validation = validate_video(Path(video_path))
        if not validation.ok:
            logger.warning("Video rejeitado na validacao: %s", validation.message)
            return (
                empty_state("Video fora dos limites aceitos.", hint=validation.message),
                "",
                None,
                None,
                None,
                None,
                [],
                {"events": []},
            )

        # Adapter: gr.Progress espera `progress(frac, desc=...)`;
        # VideoPipeline.process chama `cb(frac, desc)` posicional.
        def _progress_cb(frac: float, desc: str) -> None:
            progress(frac, desc=desc)

        try:
            events = video_pipeline.process(
                Path(video_path), progress=_progress_cb,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            logger.warning("Falha ao processar vıdeo na UI: %s", exc)
            return (
                empty_state("Falha ao processar vıdeo.", hint=str(exc)),
                "",
                None,
                None,
                None,
                None,
                [],
                {"events": []},
            )

        # --- Metricas brutas ---
        from collections import Counter

        from src.video.pose import PostureCategory, classify_posture

        total_detections = sum(len(e.detections) for e in events)
        emotions_with_face = sum(1 for e in events if e.facial_emotion is not None)
        pose_frames = sum(1 for e in events if e.pose_landmarks)
        classes_set = {d.class_name for e in events for d in e.detections}

        # Classifica postura por frame e agrega categoria predominante
        posture_counts: Counter[PostureCategory] = Counter()
        for e in events:
            if e.pose_landmarks:
                posture_counts[classify_posture(e.pose_landmarks)] += 1
        # Remove INDEFINIDO da contagem pra escolher categoria valida
        valid_postures = {k: v for k, v in posture_counts.items()
                          if k != PostureCategory.INDEFINIDO}
        dominant_posture = (
            max(valid_postures, key=valid_postures.get) if valid_postures else None
        )

        # --- Disponibilidade do MediaPipe (Py 3.14 vem com pacote reduzido) ---
        pose_available = getattr(video_pipeline.pose_estimator, "_available", True)
        # Considera disponivel se ainda nao carregou (lazy); checa apos primeira chamada
        if not video_pipeline.pose_estimator._load_attempted:
            pose_available = True

        # --- Estado da cena + decisoes do pipeline ---
        scene_type = getattr(video_pipeline, "last_scene_type", None)
        scene_value = scene_type.value if scene_type else "desconhecido"
        scene_is_consultation = scene_value == "consulta"
        scene_is_surgery = scene_value == "cirurgia"

        # --- Fonte da emocao (Azure GPT-vision ou FER local) ---
        emotion_classifier = getattr(video_pipeline, "emotion_classifier", None)
        emotion_src_name = (
            type(emotion_classifier).__name__ if emotion_classifier else "indefinido"
        )
        emotion_src_pretty = emotion_source_label(emotion_src_name)

        # --- Nivel de risco heuristico da aba (so critical se blood detectado) ---
        level = "critical" if "bleeding" in classes_set or "blood" in classes_set else "normal"

        # --- Status block ---
        status_block = (
            '<div style="display: flex; flex-direction: column; gap: 10px;">'
            '<div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">'
            f"{risk_badge(level)}"
            f"{scene_type_badge(scene_value)}"
            f'<span style="color: var(--body-text-color); font-size: 12px; opacity: 0.85;">'
            f"Emocao via: <strong>{emotion_src_pretty}</strong>"
            "</span>"
            "</div>"
            '<span style="color: var(--body-text-color); font-size: 12px; opacity: 0.7;">'
            "Heuristica da aba; a classificacao final esta em Multimodal."
            "</span>"
            "</div>"
        )

        # --- KPIs adaptados com hints especificos por cena ---
        if scene_is_consultation:
            detections_value = "0"
            detections_hint = "n/a: detector laparoscopico nao aplica"
            classes_value = "0"
            classes_hint = "sem instrumentos cirurgicos esperados"
        else:
            detections_value = str(total_detections)
            detections_hint = f"{total_detections} bbox em {len(events)} frames"
            classes_value = str(len(classes_set))
            classes_hint = (
                ", ".join(sorted(classes_set))
                if classes_set else "nenhuma classe acima do threshold"
            )

        if scene_is_surgery:
            emotions_value = "n/a"
            emotions_hint = "rosto nao visivel em campo cirurgico"
        else:
            emotions_value = str(emotions_with_face)
            emotions_hint = f"{emotions_with_face} de {len(events)} frames com face"

        if not pose_available:
            posture_value = "n/a"
            posture_hint = "MediaPipe Pose indisponivel (rodar via Docker/Py 3.12)"
        elif scene_is_surgery:
            posture_value = "n/a"
            posture_hint = "corpo nao visivel em campo cirurgico"
        elif dominant_posture is not None:
            posture_value = dominant_posture.value
            posture_hint = (
                f"{pose_frames} de {len(events)} frames; "
                f"categoria predominante via heuristica"
            )
        elif pose_frames > 0:
            posture_value = "indefinido"
            posture_hint = (
                f"{pose_frames} de {len(events)} frames com pose, "
                f"mas ombros/quadris ocultos para classificacao"
            )
        else:
            posture_value = "0"
            posture_hint = f"0 de {len(events)} frames com pose detectada"

        kpis = kpi_grid(
            [
                kpi_tile("Frames", str(len(events)), hint="amostrados pelo pipeline"),
                kpi_tile("Deteccoes", detections_value, hint=detections_hint),
                kpi_tile("Emocoes", emotions_value, hint=emotions_hint),
                kpi_tile("Postura", posture_value, hint=posture_hint),
                kpi_tile("Classes", classes_value, hint=classes_hint),
            ]
        )

        # --- Miniaturas com bboxes (lista vazia se nao houver deteccao) ---
        thumbs = build_detection_thumbnails(video_path, events, max_thumbs=4)
        has_thumbs = bool(thumbs)

        # --- Miniaturas de pose (skeleton + keypoints + bbox) ---
        pose_thumbs = build_pose_thumbnails(video_path, events, max_thumbs=4)
        has_pose_thumbs = bool(pose_thumbs)

        rows = video_events_to_windowed_rows(events, window_seconds=5.0)
        payload = {"events": [e.model_dump() for e in events[:50]]}
        timeline = build_video_timeline_plot(events)
        return (
            status_block,
            kpis,
            gr.update(value=thumbs if has_thumbs else None),
            gr.update(visible=has_thumbs),
            gr.update(value=pose_thumbs if has_pose_thumbs else None),
            gr.update(visible=has_pose_thumbs),
            timeline,
            rows,
            payload,
        )

    analyze_btn.click(
        fn=lambda: gr.update(interactive=False, value="Analisando..."),
        outputs=analyze_btn,
        queue=False,
    ).then(
        fn=_on_analyze,
        inputs=[video_input],
        outputs=[
            status_html,
            kpis_html,
            detection_thumb,
            detection_section,
            pose_thumb,
            pose_section,
            timeline_plot,
            events_table,
            raw_json,
        ],
        # show_progress="minimal" pra deixar gr.Progress do callback aparecer
        # com texto detalhado em vez do spinner generico do "full"
        show_progress="minimal",
    ).then(
        fn=lambda: gr.update(interactive=True, value="Analisar vıdeo"),
        outputs=analyze_btn,
        queue=False,
    )
