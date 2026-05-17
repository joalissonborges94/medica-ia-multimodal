"""Tema Gradio e CSS customizado para o medica-ia-multimodal.

Consolidacao do sistema de design (paleta dark indigo, tipografia Inter,
componentes) descrito em `docs/arquitetura/design.md`. Usado pelo `app.py`
e reaproveitado pelos componentes em `ui/components.py`.
"""

from __future__ import annotations

import gradio as gr

# ---------------------------------------------------------------------------
# Tokens de cor (paleta dark indigo)
# Espelham as tabelas em docs/arquitetura/design.md. Mantenha aqui a unica
# fonte de verdade em codigo; outros modulos da UI devem importar daqui.
# ---------------------------------------------------------------------------

# Backgrounds e bordas (paleta E: Linear-inspired)
BG_PRIMARY = "#0A0A0F"
BG_SURFACE = "#15151E"
BG_ELEVATED = "#262630"
BORDER = "#262630"

# Acentos
PRIMARY = "#5E6AD2"
PRIMARY_HOVER = "#4F5BC4"
PRIMARY_SUBTLE = "#9CA3F4"

# Tipografia
TEXT_PRIMARY = "#F4F4F8"
TEXT_SECONDARY = "#9CA3AF"
TEXT_TERTIARY = "#6B7280"

# Estados de alerta (background, texto)
ALERT_NORMAL_BG = "#4ADE80"
ALERT_NORMAL_FG = "#14532D"
ALERT_MODERATE_BG = "#FACC15"
ALERT_MODERATE_FG = "#422006"
ALERT_CRITICAL_BG = "#F87171"
ALERT_CRITICAL_FG = "#450A0A"

# Tipografia: famılias e fonte mono
FONT_FAMILY = ("Inter", "sans-serif")
FONT_MONO = ("JetBrains Mono", "monospace")


def get_theme() -> gr.themes.ThemeClass:
    """Retorna o tema customizado para o medica-ia-multimodal.

    Returns:
        Tema Gradio baseado em `gr.themes.Soft` com paleta dark indigo,
        fonte Inter e overrides de surfaces, botoes e inputs conforme
        especificado em `docs/arquitetura/design.md`.
    """
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        font=FONT_FAMILY,
        font_mono=FONT_MONO,
        # Variaveis de espacamento e raios elevadas no nivel de construtor.
        # Sao a fonte de verdade dos componentes nativos (gr.Group, gr.Row,
        # gr.Textbox, gr.Dataframe, etc.) e sobrescrevem o que o tema Soft
        # injeta como CSS de altissima especificidade.
        spacing_size=gr.themes.sizes.spacing_lg,
        radius_size=gr.themes.sizes.radius_lg,
    ).set(
        # ----- Cores -------------------------------------------------------
        body_background_fill=BG_PRIMARY,
        body_text_color=TEXT_PRIMARY,
        background_fill_primary=BG_PRIMARY,
        background_fill_secondary=BG_SURFACE,
        block_background_fill=BG_SURFACE,
        block_border_color=BORDER,
        # Labels dos componentes (textos como "Limite", "Audit ID", "Contexto
        # clinico"). O tema Soft injeta primary_600 no dark mode, fazendo
        # esses labels parecerem botoes. Neutralizamos para BG_ELEVATED,
        # reservando o azul indigo apenas para botoes primary.
        block_label_background_fill=BG_ELEVATED,
        block_label_background_fill_dark=BG_ELEVATED,
        block_label_text_color=TEXT_SECONDARY,
        block_label_text_color_dark=TEXT_SECONDARY,
        block_label_border_color=BORDER,
        block_title_text_color=TEXT_PRIMARY,
        border_color_primary=BORDER,
        button_primary_background_fill=PRIMARY,
        button_primary_background_fill_hover=PRIMARY_HOVER,
        button_primary_text_color="#FFFFFF",
        button_secondary_background_fill=BG_SURFACE,
        button_secondary_background_fill_hover=BG_ELEVATED,
        button_secondary_text_color=TEXT_PRIMARY,
        input_background_fill=BG_SURFACE,
        input_border_color=BORDER,
        input_border_color_focus=PRIMARY,
        # ----- Espacamento e raios (variaveis CSS reais do Gradio) ---------
        # Padding interno dos blocos (gr.Group, dataframes, plots, etc.).
        block_padding="22px",
        # Raio dos cards e dataframes.
        block_radius="14px",
        block_border_width="1px",
        # Gap entre componentes em Row/Column.
        layout_gap="20px",
        # Tipografia (define a familia consumida via var(--font)).
        block_title_text_size="14px",
        block_title_text_weight="600",
        # Inputs (textbox, number, audio, video, etc.).
        input_padding="12px 14px",
        input_radius="10px",
        # Botoes.
        button_large_text_weight="600",
        button_large_padding="12px 20px",
        button_large_radius="10px",
        button_small_text_weight="500",
        button_small_radius="8px",
    )


def get_custom_css() -> str:
    """Retorna o CSS adicional para componentes nao cobertos pelo tema base.

    Cobre: import da fonte Inter, logo do header com acento, status pill,
    badges de alerta (normal, moderado, critico), KPI tiles, section cards,
    empty states, breadcrumb de progresso, footer e ocultacao do toggle
    dark/light do painel de configuracoes do Gradio (a paleta foi pensada
    apenas para dark).

    Returns:
        Bloco CSS pronto pra passar como `css=` ao `gr.Blocks`.
    """
    return f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* ----------------------------------------------------------------------
   Estabilizacao do layout: scrollbar sempre reservada + forcar largura
   plena nos ancestrais.

   Sintoma original: a aba Audio (com menos conteudo) encolhia ~200px
   horizontalmente, fazendo o container, header e todas as secoes
   "saltarem" para o centro ao trocar de aba. Causa raiz: o custom
   element <gradio-app> nao tem display:block por default no Gradio 5,
   herdando display:inline-* que faz o elemento assumir largura
   natural do conteudo (= a aba mais estreita).
   ---------------------------------------------------------------------- */
html {{
    scrollbar-gutter: stable;
    overflow-y: scroll;
}}

body, gradio-app {{
    display: block !important;
    width: 100% !important;
}}

/* Wrapper interno que o Gradio injeta entre <gradio-app> e .gradio-container */
gradio-app .main,
gradio-app > .wrap {{
    width: 100% !important;
}}

/* ----------------------------------------------------------------------
   Container global do app
   Cantos arredondados, padding e gap entre blocos sao controlados via
   variaveis do tema em `get_theme().set(block_radius=..., layout_gap=...)`,
   nao por seletores aqui (o Gradio 5 nao expoe a classe `gr-group`).
   ---------------------------------------------------------------------- */
.gradio-container {{
    font-family: Inter, sans-serif;
    width: 100% !important;
    max-width: 1500px !important;
    margin: 0 auto !important;
    padding: 8px 32px 24px 32px !important;
    box-sizing: border-box !important;
}}

/* Cada Tab/TabItem renderizado deve esticar para a largura do container */
[role="tabpanel"],
.tabitem {{
    width: 100% !important;
}}

/* ----------------------------------------------------------------------
   Header / Logo
   ---------------------------------------------------------------------- */
.app-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 4px 22px 4px;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 22px;
}}

/* ----------------------------------------------------------------------
   Aba "Configuracoes" empurrada para a direita na barra de Tabs.
   O elem_classes "tab-config-end" propagado pelo gradio bate em multiplos
   nos da hierarquia; cobrimos ambos via :has() / atributos.
   ---------------------------------------------------------------------- */
.gradio-container .tab-nav button.tab-config-end,
.gradio-container [role="tablist"] button.tab-config-end {{
    margin-left: auto !important;
}}

/* Fallback robusto: se a classe nao bater no botao da nav, garante que a
   ultima aba (Configuracoes) cole na direita */
.gradio-container .tab-nav > button:last-of-type,
.gradio-container [role="tablist"] > button:last-of-type {{
    margin-left: auto !important;
}}

.app-header .logo {{
    font-family: Inter, sans-serif;
    font-weight: 600;
    font-size: 20px;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.01em;
}}

.app-header .logo .accent {{
    color: {PRIMARY};
}}

.app-header .status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 999px;
    padding: 4px 12px;
    color: {TEXT_SECONDARY};
    font-size: 12px;
    font-weight: 500;
}}

.app-header .status-pill .dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: {ALERT_NORMAL_BG};
    box-shadow: 0 0 8px {ALERT_NORMAL_BG};
}}

/* ----------------------------------------------------------------------
   Badges de alerta (consumidos pelas abas)
   ---------------------------------------------------------------------- */
.alert-normal, .alert-moderate, .alert-critical {{
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    display: inline-block;
}}

.alert-normal {{
    background-color: {ALERT_NORMAL_BG};
    color: {ALERT_NORMAL_FG};
}}

.alert-moderate {{
    background-color: {ALERT_MODERATE_BG};
    color: {ALERT_MODERATE_FG};
}}

.alert-critical {{
    background-color: {ALERT_CRITICAL_BG};
    color: {ALERT_CRITICAL_FG};
}}

/* ----------------------------------------------------------------------
   KPI Tiles (3 a 4 destaques no topo do resultado)
   ---------------------------------------------------------------------- */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px;
    margin: 12px 0 18px 0;
}}

.kpi-tile {{
    background: {BG_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 18px;
    transition: border-color 0.15s ease, transform 0.15s ease;
}}

.kpi-tile:hover {{
    border-color: {PRIMARY_SUBTLE};
    transform: translateY(-1px);
}}

.kpi-tile .kpi-label {{
    font-size: 11px;
    color: {TEXT_SECONDARY};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
}}

.kpi-tile .kpi-value {{
    font-size: 24px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    line-height: 1.2;
}}

.kpi-tile .kpi-hint {{
    font-size: 11px;
    color: {TEXT_TERTIARY};
    margin-top: 6px;
}}

/* ----------------------------------------------------------------------
   Section cards (envelopam grupos de widgets em cada aba)
   ---------------------------------------------------------------------- */
.section-title {{
    font-size: 16px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    margin: 2px 0 6px 0;
    display: flex;
    align-items: center;
    gap: 10px;
    letter-spacing: -0.005em;
}}

.section-title::before {{
    content: "";
    width: 3px;
    height: 16px;
    background: {PRIMARY};
    border-radius: 2px;
}}

.section-subtitle {{
    font-size: 13px;
    color: {TEXT_SECONDARY};
    margin-bottom: 16px;
    line-height: 1.55;
}}

/* ----------------------------------------------------------------------
   Tabelas readonly (aba Audit, aba Video etc.): colapsar espaco
   reservado pra grow dinamico quando ja se sabe o numero de linhas
   atual (interactive=False). Gradio Dataframe reserva ~3-4 linhas
   vazias por baixo "pra crescer". `height: auto` + `max-height` controla
   via CSS. Aplicar `elem_classes="compact-table"` no `gr.Dataframe`.
   ---------------------------------------------------------------------- */
.gradio-container .audit-table .table-wrap,
.gradio-container .audit-table .svelte-virtual-table-viewport,
.gradio-container .compact-table .table-wrap,
.gradio-container .compact-table .svelte-virtual-table-viewport {{
    height: auto !important;
    max-height: 520px;
}}

/* ----------------------------------------------------------------------
   Content box: emula o visual do gr.Textbox para texto/markdown denso.

   Insight: gr.Textbox renderiza com box interno (background + border +
   border-radius + padding) que descola o conteudo da borda do card pai.
   Markdowns sem isso ficavam "colados". Replicamos o visual de input
   aqui pra ter consistencia entre Contexto clinico (textbox) e relatorio,
   anomalia, diretrizes, etc. (markdown).

   `content-box` deve ser passado como `elem_classes` em uma gr.Column
   wrapper de qualquer gr.Markdown, gr.HTML ou gr.JSON que precise do
   visual de "input lido".

   Variaveis do tema (input_background_fill, input_border_color,
   input_radius) garantem alinhamento com o resto da UI.
   ---------------------------------------------------------------------- */
.gradio-container .content-box {{
    background: var(--input-background-fill, {BG_SURFACE});
    border: 1px solid var(--input-border-color, {BORDER});
    border-radius: 10px;
    padding: 14px 20px !important;
    margin-top: 4px;
}}

/* Remove backgrounds duplicados de filhos imediatos (ex.: gr.Markdown
   default tem o seu proprio .block bg que pinta sobre o content-box) */
.gradio-container .content-box > .block,
.gradio-container .content-box > * {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}}

/* ----------------------------------------------------------------------
   Markdown dentro de cards (Relatorio clinico, Resumo de anomalia, etc.)
   Tipografia: line-height, espaco vertical entre blocos (h1/h2/h3/p/ul)
   e listas indentadas pra leitura confortavel.
   ---------------------------------------------------------------------- */
.gradio-container .prose,
.gradio-container .md {{
    line-height: 1.65;
    max-width: 100%;
}}

.gradio-container .prose h1,
.gradio-container .md h1 {{
    font-size: 22px;
    margin: 14px 0 10px 0;
    letter-spacing: -0.01em;
}}

.gradio-container .prose h2,
.gradio-container .md h2 {{
    font-size: 17px;
    margin: 22px 0 8px 0;
    letter-spacing: -0.005em;
}}

.gradio-container .prose h3,
.gradio-container .md h3 {{
    font-size: 15px;
    margin: 16px 0 6px 0;
}}

.gradio-container .prose p,
.gradio-container .md p {{
    margin: 6px 0 10px 0;
}}

.gradio-container .prose ul,
.gradio-container .md ul,
.gradio-container .prose ol,
.gradio-container .md ol {{
    padding-left: 22px;
    margin: 6px 0 12px 0;
}}

.gradio-container .prose li,
.gradio-container .md li {{
    margin: 4px 0;
    line-height: 1.55;
}}

/* ----------------------------------------------------------------------
   Empty state (placeholder informativo das areas de saida)
   ---------------------------------------------------------------------- */
.empty-state {{
    padding: 24px 20px;
    border: 1px dashed {BORDER};
    border-radius: 12px;
    background: transparent;
    color: {TEXT_SECONDARY};
    font-size: 13px;
    line-height: 1.55;
    text-align: center;
    margin: 8px 0;
}}

.empty-state .empty-state-title {{
    font-weight: 600;
    color: {TEXT_PRIMARY};
    margin-bottom: 4px;
    font-size: 13px;
}}

.empty-state .empty-state-hint {{
    color: {TEXT_TERTIARY};
    font-size: 12px;
}}

/* ----------------------------------------------------------------------
   Breadcrumb de progresso (Upload -> Processar -> Resultado)
   ---------------------------------------------------------------------- */
.breadcrumb {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0 14px 0;
    font-size: 12px;
    color: {TEXT_SECONDARY};
    margin-bottom: 4px;
}}

.breadcrumb .step {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid {BORDER};
    background: {BG_SURFACE};
}}

.breadcrumb .step.active {{
    border-color: {PRIMARY};
    color: {PRIMARY_SUBTLE};
    background: rgba(99, 102, 241, 0.08);
}}

.breadcrumb .step .num {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: {BG_ELEVATED};
    font-size: 10px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}

.breadcrumb .step.active .num {{
    background: {PRIMARY};
    color: #FFFFFF;
}}

.breadcrumb .sep {{
    color: {TEXT_TERTIARY};
    font-size: 11px;
}}

/* ----------------------------------------------------------------------
   Footer
   ---------------------------------------------------------------------- */
.app-footer {{
    margin-top: 28px;
    padding: 16px 4px 8px 4px;
    border-top: 1px solid {BORDER};
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: {TEXT_TERTIARY};
    font-size: 11px;
}}

.app-footer a {{
    color: {TEXT_SECONDARY};
    text-decoration: none;
    border-bottom: 1px dotted {TEXT_TERTIARY};
}}

.app-footer a:hover {{
    color: {PRIMARY_SUBTLE};
    border-bottom-color: {PRIMARY_SUBTLE};
}}

/* ----------------------------------------------------------------------
   Ocultar configuracoes do Gradio que nao se aplicam ao app:
   - Toggle dark/light (a paleta foi desenhada apenas para dark)
   ---------------------------------------------------------------------- */
button[aria-label="Toggle dark mode"],
button[title="Toggle dark mode"],
button[aria-label*="Settings" i],
button[aria-label*="theme" i],
.theme-toggle,
.theme-toggle-wrap,
gradio-app .theme-toggle-wrap {{
    display: none !important;
}}

/* ----------------------------------------------------------------------
   Botao "Recarregar" no header da aba Auditoria (variant=secondary, sm)
   Compacto, alinhado pelo topo (mesma linha do section-title).
   ---------------------------------------------------------------------- */
.audit-recarregar {{
    align-self: flex-start !important;
    padding-top: 2px !important;
    padding-right: 4px !important;
    display: flex !important;
    justify-content: flex-end !important;
}}

.audit-recarregar button {{
    width: auto !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    padding: 6px 16px !important;
    height: auto !important;
    min-height: 32px !important;
}}

/* ----------------------------------------------------------------------
   Footer do Gradio ("Construido com Gradio" + botao "Configuracoes")
   Removido para nao competir com nosso .app-footer (versao + repo + MIT).
   ---------------------------------------------------------------------- */
gradio-app footer,
.gradio-container footer,
gradio-app .built-with,
gradio-app .settings-buttons,
gradio-app .show-api,
.contain > div.svelte-1ipelgc {{
    display: none !important;
}}
"""
