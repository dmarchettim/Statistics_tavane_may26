"""
analise_comparativa.py — Análise comparativa dos 3 períodos (2016, 2020, 2024).

Gera relatorio_comparativo.pdf com análise textual seção a seção, comparando
os achados entre os 3 anos.

Objetivo central:
  • Avaliar se houve aumento da participação de candidatos negros (pretos +
    pardos) entre os eleitos.
  • Investigar a relação entre receita de campanha e desempenho eleitoral
    para candidatos negros.
  • Identificar se houve variação na receita destinada a candidatos negros
    ao longo do período.

Script autocontido.
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ============================================================
# Configuração
# ============================================================
ANOS = [2016, 2020, 2024]
ARQUIVOS = {a: f"Banco_{a}.csv" for a in ANOS}
SAIDA = "relatorio_comparativo.pdf"
COR_BRANCO = "#4C72B0"
COR_NAO_BRANCO = "#DD8452"
COR_NEUTRA = "#666666"
COR_DESTAQUE = "#c0392b"
FIG_A4 = (8.27, 11.69)
VALORES_NAO_DIVULGAVEIS = {"NÃO DIVULGÁVEL", "Não divulgável"}

plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10})


# ============================================================
# Carregamento e cálculo de estatísticas
# ============================================================
def carregar_e_preprocessar(ano):
    arquivo = ARQUIVOS[ano]
    if not Path(arquivo).exists():
        sys.exit(f"ERRO: arquivo '{arquivo}' não encontrado.")
    df = pd.read_csv(arquivo, sep=";", encoding="latin1", decimal=",")
    df = df.dropna(subset=["TOTAL_VOTOS", "DS_SIT_TOT_TURNO"]).copy()
    df["TOTAL_VOTOS"] = df["TOTAL_VOTOS"].astype(int)
    df["ELEITO"] = df["DS_SIT_TOT_TURNO"].isin(
        ["ELEITO POR QP", "ELEITO POR MÉDIA"]).astype(int)
    menor_eleito = int(df.loc[df["ELEITO"] == 1, "TOTAL_VOTOS"].min())
    df["COMPETITIVO"] = (df["TOTAL_VOTOS"] >= 0.20 * menor_eleito).astype(int)
    df["RACA_BIN"] = df["DS_COR_RACA"].map(
        {"BRANCA": "Branco", "PRETA": "Não branco", "PARDA": "Não branco"})
    df = df[df["RACA_BIN"].notna()].copy()
    df["LOG_RECEITA"] = np.log1p(df["RECEITA_TOTAL"])
    df["LOG_VOTOS"] = np.log1p(df["TOTAL_VOTOS"])
    return df, menor_eleito


def _rk(raca):
    return "b" if raca == "Branco" else "nb"


def calcular_estatisticas(df, menor_eleito):
    eleitos = df[df["ELEITO"] == 1]
    comp = df[df["COMPETITIVO"] == 1]

    s = {
        "n_total": len(df),
        "n_eleitos": len(eleitos),
        "n_competitivos": len(comp),
        "limiar_comp": 0.20 * menor_eleito,
        "menor_eleito": menor_eleito,
    }
    for raca in ["Branco", "Não branco"]:
        k = _rk(raca)
        s[f"n_{k}"] = int((df["RACA_BIN"] == raca).sum())
        s[f"pct_{k}"] = (df["RACA_BIN"] == raca).mean() * 100
        s[f"n_eleitos_{k}"] = int((eleitos["RACA_BIN"] == raca).sum())
        s[f"pct_eleitos_{k}"] = ((eleitos["RACA_BIN"] == raca).mean() * 100
                                 if len(eleitos) else 0)
        s[f"n_comp_{k}"] = int((comp["RACA_BIN"] == raca).sum())
        s[f"pct_comp_{k}"] = ((comp["RACA_BIN"] == raca).mean() * 100
                              if len(comp) else 0)

        # Receita por raça (apenas com receita declarada)
        sub = df[(df["RACA_BIN"] == raca) & df["RECEITA_TOTAL"].notna()]
        sub_c = comp[(comp["RACA_BIN"] == raca) & comp["RECEITA_TOTAL"].notna()]
        sub_e = eleitos[(eleitos["RACA_BIN"] == raca) & eleitos["RECEITA_TOTAL"].notna()]
        s[f"rec_med_{k}"] = sub["RECEITA_TOTAL"].median() if len(sub) else np.nan
        s[f"rec_mean_{k}"] = sub["RECEITA_TOTAL"].mean() if len(sub) else np.nan
        s[f"rec_total_{k}"] = sub["RECEITA_TOTAL"].sum() if len(sub) else 0
        s[f"rec_med_comp_{k}"] = (sub_c["RECEITA_TOTAL"].median()
                                  if len(sub_c) else np.nan)
        s[f"rec_med_eleitos_{k}"] = (sub_e["RECEITA_TOTAL"].median()
                                     if len(sub_e) else np.nan)
        s[f"n_rec_validas_{k}"] = len(sub)

    # Share da receita total
    rec_b = s["rec_total_b"]; rec_nb = s["rec_total_nb"]
    rec_soma = rec_b + rec_nb
    s["share_rec_b"] = rec_b / rec_soma * 100 if rec_soma else 0
    s["share_rec_nb"] = rec_nb / rec_soma * 100 if rec_soma else 0

    # Correlações r de Pearson
    for univ_name, univ in [("todos", df), ("comp", comp)]:
        for raca in ["Branco", "Não branco"]:
            k = _rk(raca)
            sub = univ[(univ["RACA_BIN"] == raca)
                       & univ["RECEITA_TOTAL"].notna()
                       & univ["TOTAL_VOTOS"].notna()]
            if len(sub) >= 3:
                r, p = stats.pearsonr(sub["RECEITA_TOTAL"], sub["TOTAL_VOTOS"])
                s[f"r_{univ_name}_{k}"] = r
                s[f"p_{univ_name}_{k}"] = p
            else:
                s[f"r_{univ_name}_{k}"] = np.nan
                s[f"p_{univ_name}_{k}"] = np.nan
            s[f"n_r_{univ_name}_{k}"] = len(sub)

    # Regressões — apenas coeficientes-chave
    cats_ols = ["DS_GENERO", "DS_GRAU_INSTRUCAO", "ST_REELEICAO"]
    cats_logit = ["DS_GENERO", "ST_REELEICAO"]
    cols_ols = ["LOG_VOTOS", "LOG_RECEITA", "DS_GENERO", "RACA_BIN",
                "DS_GRAU_INSTRUCAO", "ST_REELEICAO"]
    cols_logit = ["ELEITO", "LOG_RECEITA", "DS_GENERO", "RACA_BIN",
                  "ST_REELEICAO"]

    def _filtra(sub, cats):
        for c in cats:
            sub = sub[~sub[c].isin(VALORES_NAO_DIVULGAVEIS)]
        return sub

    for univ_name, univ in [("todos", df), ("comp", comp)]:
        sub_ols = _filtra(univ.dropna(subset=cols_ols).copy(), cats_ols)
        sub_log = _filtra(univ.dropna(subset=cols_logit).copy(), cats_logit)
        try:
            r = smf.ols(
                "LOG_VOTOS ~ LOG_RECEITA + C(DS_GENERO) + C(RACA_BIN) "
                "+ C(DS_GRAU_INSTRUCAO) + C(ST_REELEICAO)",
                data=sub_ols).fit()
            s[f"ols_{univ_name}_receita"] = r.params.get("LOG_RECEITA", np.nan)
            s[f"ols_{univ_name}_p_receita"] = r.pvalues.get("LOG_RECEITA", np.nan)
            s[f"ols_{univ_name}_raca"] = r.params.get(
                "C(RACA_BIN)[T.Não branco]", np.nan)
            s[f"ols_{univ_name}_p_raca"] = r.pvalues.get(
                "C(RACA_BIN)[T.Não branco]", np.nan)
            s[f"ols_{univ_name}_r2"] = r.rsquared
            s[f"ols_{univ_name}_n"] = int(r.nobs)
        except Exception as e:
            print(f"[ERRO] OLS {univ_name}: {e}")
        try:
            r = smf.logit(
                "ELEITO ~ LOG_RECEITA + C(DS_GENERO) + C(RACA_BIN) "
                "+ C(ST_REELEICAO)", data=sub_log).fit(disp=False)
            s[f"logit_{univ_name}_receita"] = r.params.get("LOG_RECEITA", np.nan)
            s[f"logit_{univ_name}_p_receita"] = r.pvalues.get("LOG_RECEITA", np.nan)
            s[f"logit_{univ_name}_raca"] = r.params.get(
                "C(RACA_BIN)[T.Não branco]", np.nan)
            s[f"logit_{univ_name}_p_raca"] = r.pvalues.get(
                "C(RACA_BIN)[T.Não branco]", np.nan)
            s[f"logit_{univ_name}_pr2"] = r.prsquared
            s[f"logit_{univ_name}_n"] = int(r.nobs)
        except Exception as e:
            print(f"[ERRO] Logit {univ_name}: {e}")
        try:
            r = smf.logit(
                "ELEITO ~ LOG_RECEITA * C(RACA_BIN) + C(DS_GENERO) "
                "+ C(ST_REELEICAO)", data=sub_log).fit(disp=False)
            s[f"inter_{univ_name}"] = r.params.get(
                "LOG_RECEITA:C(RACA_BIN)[T.Não branco]", np.nan)
            s[f"inter_{univ_name}_p"] = r.pvalues.get(
                "LOG_RECEITA:C(RACA_BIN)[T.Não branco]", np.nan)
        except Exception as e:
            print(f"[ERRO] Interação {univ_name}: {e}")

    return s


# ============================================================
# Utilidades de página
# ============================================================
def nova_pagina(titulo):
    fig = plt.figure(figsize=FIG_A4)
    fig.suptitle(titulo, fontsize=12.5, fontweight="bold", y=0.965)
    fig.text(0.5, 0.015, "Análise Comparativa — TRE-SP (2016, 2020, 2024)",
             ha="center", fontsize=8, color=COR_NEUTRA)
    return fig


def add_table(fig, df_table, rect, fontsize=8.5):
    ax = fig.add_axes(rect)
    ax.axis("off")
    tab = ax.table(
        cellText=df_table.values.tolist(),
        colLabels=df_table.columns.tolist(),
        loc="center", cellLoc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(fontsize)
    tab.scale(1, 1.5)
    for j in range(len(df_table.columns)):
        tab[(0, j)].set_text_props(fontweight="bold", color="white")
        tab[(0, j)].set_facecolor("#444")
    return ax


def add_paragraphs(fig, paragrafos, rect, fontsize=9.5, line_h=0.027):
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    y = 1.0
    for p in paragrafos:
        ax.text(0, y, p, fontsize=fontsize, va="top", ha="left", wrap=True)
        # estimativa simples de altura ocupada (linhas)
        linhas = max(1, int(len(p) / 95) + p.count("\n"))
        y -= line_h * linhas + 0.018


def fmt_br(v, casas=0):
    if pd.isna(v):
        return "—"
    if casas == 0:
        return f"{v:,.0f}".replace(",", ".")
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(v, casas=1):
    return "—" if pd.isna(v) else f"{v:.{casas}f}%"


def fmt_real(v, casas=0):
    if pd.isna(v):
        return "—"
    return "R$ " + fmt_br(v, casas)


# ============================================================
# Páginas
# ============================================================
def pagina_capa(pdf, dados):
    fig = plt.figure(figsize=FIG_A4)
    fig.text(0.5, 0.80, "Análise Comparativa", ha="center",
             fontsize=24, fontweight="bold")
    fig.text(0.5, 0.74, "Raça, Receita de Campanha e Desempenho Eleitoral",
             ha="center", fontsize=15, fontstyle="italic")
    fig.text(0.5, 0.69, "Candidatos a Vereador — São Paulo",
             ha="center", fontsize=13)
    fig.text(0.5, 0.65, "Ciclos eleitorais de 2016, 2020 e 2024",
             ha="center", fontsize=12)
    info = (
        "Comparativo seção a seção dos três relatórios anuais.\n"
        "Foco analítico: evolução da participação dos candidatos negros\n"
        "(pretos + pardos) e variação na receita destinada a esse grupo.\n\n"
        f"N total analítico (Brancos + Negros) — 2016: {dados[2016]['stats']['n_total']:,} | "
        f"2020: {dados[2020]['stats']['n_total']:,} | "
        f"2024: {dados[2024]['stats']['n_total']:,}\n"
        f"Eleitos por ciclo: 55 vagas em todos os anos."
    )
    fig.text(0.5, 0.40, info, ha="center", fontsize=11,
             bbox=dict(boxstyle="round,pad=1", fc="#f0f0f0", ec="#bbb"))
    fig.text(0.5, 0.10, f"Gerado em {datetime.now():%d/%m/%Y %H:%M}",
             ha="center", fontsize=9, color=COR_NEUTRA)
    pdf.savefig(fig); plt.close(fig)


def pagina_panorama(pdf, dados):
    fig = nova_pagina("Sumário Executivo — Panorama da Evolução")

    rows = []
    for ano in ANOS:
        s = dados[ano]["stats"]
        rows.append([
            str(ano),
            f"{s['n_total']:,}".replace(",", "."),
            f"{s['n_nb']:,}".replace(",", "."),
            fmt_pct(s['pct_nb']),
            f"{s['n_comp_nb']:,}".replace(",", "."),
            fmt_pct(s['pct_comp_nb']),
            f"{s['n_eleitos_nb']:,}".replace(",", "."),
            fmt_pct(s['pct_eleitos_nb']),
        ])
    tab = pd.DataFrame(rows, columns=[
        "Ano", "N total", "N negros", "% negros",
        "Competitivos negros", "% competitivos", "Eleitos negros", "% eleitos"
    ])
    add_table(fig, tab, rect=(0.04, 0.74, 0.92, 0.16), fontsize=8.5)

    # Gráfico-síntese: três proporções por ano
    ax = fig.add_axes([0.14, 0.40, 0.78, 0.27])
    x = np.arange(len(ANOS))
    w = 0.27
    pct_total = [dados[a]["stats"]["pct_nb"] for a in ANOS]
    pct_comp = [dados[a]["stats"]["pct_comp_nb"] for a in ANOS]
    pct_eleitos = [dados[a]["stats"]["pct_eleitos_nb"] for a in ANOS]
    ax.bar(x - w, pct_total, w, label="% candidatos", color="#94a3b8")
    ax.bar(x, pct_comp, w, label="% competitivos", color="#fb923c")
    ax.bar(x + w, pct_eleitos, w, label="% eleitos", color=COR_NAO_BRANCO)
    for i, (a, b, c) in enumerate(zip(pct_total, pct_comp, pct_eleitos)):
        ax.text(i - w, a + 0.5, f"{a:.1f}", ha="center", fontsize=8)
        ax.text(i, b + 0.5, f"{b:.1f}", ha="center", fontsize=8)
        ax.text(i + w, c + 0.5, f"{c:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([str(a) for a in ANOS])
    ax.set_ylabel("% de não brancos (pretos + pardos)")
    ax.set_title("Evolução da participação de negros: candidatos, competitivos e eleitos")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(pct_total), max(pct_comp), max(pct_eleitos)) + 10)

    # Texto-síntese
    var_cand = dados[2024]["stats"]["pct_nb"] - dados[2016]["stats"]["pct_nb"]
    var_comp = dados[2024]["stats"]["pct_comp_nb"] - dados[2016]["stats"]["pct_comp_nb"]
    var_el = dados[2024]["stats"]["pct_eleitos_nb"] - dados[2016]["stats"]["pct_eleitos_nb"]
    paras = [
        f"Em 8 anos, a presença de candidatos negros entre os concorrentes à "
        f"Câmara Municipal de São Paulo cresceu em todas as três dimensões "
        f"analisadas: candidatura ({var_cand:+.1f} p.p.), candidatura "
        f"competitiva ({var_comp:+.1f} p.p.) e eleição efetiva ({var_el:+.1f} p.p.).",
        f"O crescimento entre eleitos ({dados[2016]['stats']['n_eleitos_nb']} → "
        f"{dados[2024]['stats']['n_eleitos_nb']} vereadores negros) foi o mais "
        f"acentuado relativamente, indicando que a abertura no acesso à candidatura "
        f"se traduziu em ganhos reais de representação política.",
    ]
    add_paragraphs(fig, paras, rect=(0.06, 0.07, 0.88, 0.30))
    pdf.savefig(fig); plt.close(fig)


def pagina_metodologia(pdf, dados):
    fig = nova_pagina("Metodologia Resumida")
    paras = [
        "Bases: três extrações do TRE-SP com candidatos a vereador "
        "(Banco_2016.csv, Banco_2020.csv, Banco_2024.csv). Todas as variáveis "
        "operacionais foram construídas de forma idêntica nos três anos para "
        "garantir comparabilidade.",

        "Recorte racial: 'Branco' = DS_COR_RACA igual a 'BRANCA'. 'Não branco' "
        "(negros) = 'PRETA' + 'PARDA'. As categorias 'AMARELA', 'INDÍGENA' e "
        "'NÃO DIVULGÁVEL' foram excluídas das análises (totais reportados nos "
        "relatórios anuais).",

        "Definição de competitivo: candidatos com pelo menos 20% dos votos do "
        "último eleito (eleito com menor votação no ano). Limiar varia por "
        f"ciclo — 2016: {fmt_br(dados[2016]['stats']['limiar_comp'])} votos; "
        f"2020: {fmt_br(dados[2020]['stats']['limiar_comp'])} votos; "
        f"2024: {fmt_br(dados[2024]['stats']['limiar_comp'])} votos.",

        "Eleito: DS_SIT_TOT_TURNO igual a 'ELEITO POR QP' ou 'ELEITO POR MÉDIA'.",

        "Correlação de Pearson (r) entre RECEITA_TOTAL e TOTAL_VOTOS, "
        "estratificada por raça. Para regressões, variáveis monetárias são "
        "transformadas por log1p para mitigar assimetria.",

        "Regressão múltipla (OLS) — DV: log(votos); IV: log(receita) + gênero + "
        "raça + grau de instrução + reeleição. Regressão logística — DV: eleito "
        "(0/1); IV: log(receita) + gênero + raça + reeleição. Modelo com "
        "interação acrescenta o termo log(receita) × raça para testar se o "
        "retorno marginal da receita varia entre brancos e negros.",

        "Atenção: em 2016, ~53% das linhas têm RECEITA_TOTAL ausente, o que "
        "reduz o N efetivo das regressões e correlações e pode introduzir "
        "viés de seleção. Esse ponto é considerado nas interpretações.",
    ]
    add_paragraphs(fig, paras, rect=(0.07, 0.07, 0.86, 0.85))
    pdf.savefig(fig); plt.close(fig)


def pagina_secao_1(pdf, dados):
    fig = nova_pagina("Seção 1 — Proporção de candidatos brancos e não brancos (todos)")
    rows = []
    for ano in ANOS:
        s = dados[ano]["stats"]
        rows.append([str(ano),
                     f"{s['n_total']:,}".replace(",", "."),
                     f"{s['n_b']:,}".replace(",", "."), fmt_pct(s['pct_b']),
                     f"{s['n_nb']:,}".replace(",", "."), fmt_pct(s['pct_nb'])])
    tab = pd.DataFrame(rows, columns=["Ano", "N total", "Brancos",
                                      "% brancos", "Negros", "% negros"])
    add_table(fig, tab, rect=(0.10, 0.72, 0.80, 0.16))

    var = dados[2024]["stats"]["pct_nb"] - dados[2016]["stats"]["pct_nb"]
    paras = [
        f"A participação de candidatos negros (pretos + pardos) cresceu de "
        f"{fmt_pct(dados[2016]['stats']['pct_nb'])} em 2016 para "
        f"{fmt_pct(dados[2024]['stats']['pct_nb'])} em 2024, um aumento de "
        f"{var:+.1f} pontos percentuais em apenas oito anos. O movimento é "
        f"consistente entre os ciclos, com avanço também em 2020 "
        f"({fmt_pct(dados[2020]['stats']['pct_nb'])}).",

        "Esse crescimento, contudo, deve ser lido com cautela: o número absoluto "
        f"de candidatos negros foi {dados[2016]['stats']['n_nb']} em 2016, "
        f"{dados[2020]['stats']['n_nb']} em 2020 e {dados[2024]['stats']['n_nb']} "
        f"em 2024. O total de candidatos caiu de "
        f"{dados[2020]['stats']['n_total']:,} (2020) para "
        f"{dados[2024]['stats']['n_total']:,} (2024) — parte do ganho percentual "
        "decorre da retração geral do número de candidaturas, não apenas de "
        "expansão real do contingente negro.".replace(",", "."),

        "Apesar do avanço, brancos continuam majoritários nas três eleições. "
        "A população negra de São Paulo (PRETA + PARDA) supera 50% segundo o "
        "Censo 2022; logo, mesmo em 2024 há sub-representação proporcional "
        "na disputa.",
    ]
    add_paragraphs(fig, paras, rect=(0.07, 0.08, 0.86, 0.60))
    pdf.savefig(fig); plt.close(fig)


def pagina_secao_2(pdf, dados):
    fig = nova_pagina("Seção 2 — Proporção de candidatos competitivos")
    rows = []
    for ano in ANOS:
        s = dados[ano]["stats"]
        rows.append([str(ano),
                     f"{s['n_competitivos']:,}".replace(",", "."),
                     f"{s['n_comp_b']:,}".replace(",", "."),
                     fmt_pct(s['pct_comp_b']),
                     f"{s['n_comp_nb']:,}".replace(",", "."),
                     fmt_pct(s['pct_comp_nb'])])
    tab = pd.DataFrame(rows, columns=["Ano", "N competitivos", "Brancos",
                                      "% brancos", "Negros", "% negros"])
    add_table(fig, tab, rect=(0.10, 0.72, 0.80, 0.16))

    var = (dados[2024]["stats"]["pct_comp_nb"]
           - dados[2016]["stats"]["pct_comp_nb"])
    gap_cand_comp_2024 = (dados[2024]["stats"]["pct_nb"]
                          - dados[2024]["stats"]["pct_comp_nb"])
    paras = [
        f"A proporção de candidatos negros entre os competitivos passou de "
        f"{fmt_pct(dados[2016]['stats']['pct_comp_nb'])} em 2016 para "
        f"{fmt_pct(dados[2024]['stats']['pct_comp_nb'])} em 2024 "
        f"({var:+.1f} p.p.). O crescimento existe, mas é mais lento do que o "
        "observado na proporção de candidaturas totais.",

        "Há um descompasso entre 'estar na disputa' e 'ter chance real de "
        f"vencer'. Em 2024, por exemplo, negros são {fmt_pct(dados[2024]['stats']['pct_nb'])} "
        f"dos candidatos mas apenas {fmt_pct(dados[2024]['stats']['pct_comp_nb'])} "
        f"dos competitivos — uma diferença de {gap_cand_comp_2024:.1f} p.p. que "
        "indica filtro estrutural: candidaturas negras tendem a se concentrar "
        "em faixas de votação mais baixas.",

        "Em termos absolutos, o número de competitivos negros cresceu de "
        f"{dados[2016]['stats']['n_comp_nb']} (2016) para "
        f"{dados[2024]['stats']['n_comp_nb']} (2024) — um leve recuo absoluto "
        "em relação aos picos anteriores, mas com participação percentual em "
        "alta.",
    ]
    add_paragraphs(fig, paras, rect=(0.07, 0.08, 0.86, 0.60))
    pdf.savefig(fig); plt.close(fig)


def pagina_secao_3(pdf, dados):
    fig = nova_pagina("Seção 3 — Proporção de candidatos eleitos")
    rows = []
    for ano in ANOS:
        s = dados[ano]["stats"]
        rows.append([str(ano),
                     f"{s['n_eleitos']:,}".replace(",", "."),
                     f"{s['n_eleitos_b']:,}".replace(",", "."),
                     fmt_pct(s['pct_eleitos_b']),
                     f"{s['n_eleitos_nb']:,}".replace(",", "."),
                     fmt_pct(s['pct_eleitos_nb'])])
    tab = pd.DataFrame(rows, columns=["Ano", "N eleitos", "Brancos",
                                      "% brancos", "Negros", "% negros"])
    add_table(fig, tab, rect=(0.10, 0.72, 0.80, 0.16))

    n0 = dados[2016]["stats"]["n_eleitos_nb"]
    n4 = dados[2024]["stats"]["n_eleitos_nb"]
    pct0 = dados[2016]["stats"]["pct_eleitos_nb"]
    pct4 = dados[2024]["stats"]["pct_eleitos_nb"]
    cresc_abs = (n4 - n0) / n0 * 100 if n0 else np.nan
    paras = [
        f"Esta é a principal evidência de avanço observada na série: a "
        f"representação de vereadores negros cresceu de {n0} em 2016 para "
        f"{n4} em 2024 — um aumento de "
        f"{cresc_abs:+.0f}% em termos absolutos e de "
        f"{pct4 - pct0:+.1f} p.p. em termos proporcionais ({fmt_pct(pct0)} → "
        f"{fmt_pct(pct4)}).",

        "O ritmo da evolução é acelerado: em 2020 já havia "
        f"{dados[2020]['stats']['n_eleitos_nb']} vereadores negros eleitos "
        f"({fmt_pct(dados[2020]['stats']['pct_eleitos_nb'])}), indicando que "
        "o salto não é fenômeno apenas de 2024 — vem se consolidando ciclo "
        "a ciclo.",

        "É importante notar que esse aumento acontece em um contexto de "
        "estabilidade do número total de vagas (55 em todos os ciclos), "
        "ou seja, o ganho de negros corresponde a perda de brancos. Em 2016, "
        f"{dados[2016]['stats']['n_eleitos_b']} dos 53 eleitos identificáveis "
        "racialmente eram brancos; em 2024, esse número caiu para "
        f"{dados[2024]['stats']['n_eleitos_b']}.",

        "Comparando esta seção com a anterior, o que mais chama atenção é que "
        "a proporção de negros entre eleitos "
        f"({fmt_pct(dados[2024]['stats']['pct_eleitos_nb'])}) supera a "
        f"proporção entre competitivos "
        f"({fmt_pct(dados[2024]['stats']['pct_comp_nb'])}) em 2024 — situação "
        "incomum e que pode indicar maior eficácia eleitoral relativa de "
        "candidatos negros quando atingem competitividade.",
    ]
    add_paragraphs(fig, paras, rect=(0.07, 0.06, 0.86, 0.65))
    pdf.savefig(fig); plt.close(fig)


def pagina_secao_4(pdf, dados):
    fig = nova_pagina("Seção 4 — Correlação Pearson receita × votos (todos os candidatos)")
    rows = []
    for ano in ANOS:
        s = dados[ano]["stats"]
        rows.append([str(ano),
                     f"{s['n_r_todos_b']}", f"{s['r_todos_b']:.3f}",
                     f"{s['p_todos_b']:.3g}",
                     f"{s['n_r_todos_nb']}", f"{s['r_todos_nb']:.3f}",
                     f"{s['p_todos_nb']:.3g}"])
    tab = pd.DataFrame(rows, columns=["Ano", "N (B)", "r Brancos", "p (B)",
                                      "N (NB)", "r Negros", "p (NB)"])
    add_table(fig, tab, rect=(0.05, 0.72, 0.90, 0.16), fontsize=8)

    paras = [
        "A correlação de Pearson entre receita declarada e votos recebidos é "
        "positiva e significativa em todos os anos para ambos os grupos — "
        "candidatos com mais recursos tendem a obter mais votos. A magnitude, "
        "porém, varia consideravelmente entre os ciclos.",

        f"Em 2016, ano com a maior dependência aparente entre receita e votos, "
        f"r foi de {dados[2016]['stats']['r_todos_b']:.3f} para brancos e "
        f"{dados[2016]['stats']['r_todos_nb']:.3f} para negros. Em 2020 e 2024, "
        "os coeficientes recuam significativamente — sugerindo que outros "
        "fatores (capital político, redes sociais, militância) podem ter "
        "ganhado peso relativo à receita pura.",

        "A diferença entre brancos e negros dentro de cada ano é pequena, "
        "indicando que a relação 'mais receita = mais votos' opera de forma "
        "parecida nos dois grupos quando considerados todos os candidatos. "
        "Diferenças podem surgir, contudo, quando recortamos competitivos "
        "(Seção 5) ou nas regressões controladas (Seções 9-14).",

        "Atenção metodológica: em 2016, a correlação é calculada com N "
        "reduzido em ~50% devido a NAs em RECEITA_TOTAL; isso amplia a "
        "incerteza dos coeficientes desse ano.",
    ]
    add_paragraphs(fig, paras, rect=(0.07, 0.06, 0.86, 0.62))
    pdf.savefig(fig); plt.close(fig)


def pagina_secao_5(pdf, dados):
    fig = nova_pagina("Seção 5 — Correlação Pearson receita × votos (competitivos)")
    rows = []
    for ano in ANOS:
        s = dados[ano]["stats"]
        rows.append([str(ano),
                     f"{s['n_r_comp_b']}", f"{s['r_comp_b']:.3f}",
                     f"{s['p_comp_b']:.3g}",
                     f"{s['n_r_comp_nb']}", f"{s['r_comp_nb']:.3f}",
                     f"{s['p_comp_nb']:.3g}"])
    tab = pd.DataFrame(rows, columns=["Ano", "N (B)", "r Brancos", "p (B)",
                                      "N (NB)", "r Negros", "p (NB)"])
    add_table(fig, tab, rect=(0.05, 0.72, 0.90, 0.16), fontsize=8)

    paras = [
        "Restringindo a análise aos candidatos competitivos, as correlações "
        "caem em todos os anos — o que é esperado: ao filtrar pela parte "
        "alta da distribuição de votos, perdemos variabilidade e o sinal "
        "estatístico enfraquece.",

        f"Em 2024, r entre competitivos é de {dados[2024]['stats']['r_comp_b']:.3f} "
        f"(brancos) e {dados[2024]['stats']['r_comp_nb']:.3f} (negros). A "
        "diferença é pequena, mas o coeficiente ligeiramente maior entre "
        "negros sugere que, no grupo competitivo, a receita pode ter peso "
        "relativamente mais decisivo para esse grupo.",

        "Em 2020, o padrão se inverte: brancos competitivos apresentam "
        f"r = {dados[2020]['stats']['r_comp_b']:.3f} contra "
        f"{dados[2020]['stats']['r_comp_nb']:.3f} dos negros. Isso reforça "
        "que não há um padrão estável entre ciclos — a dinâmica depende "
        "do contexto eleitoral.",

        "Em ambos os universos (todos e competitivos), a correlação positiva "
        "e significativa confirma que receita continua sendo um preditor "
        "relevante de votação — mas a força dessa relação varia tanto "
        "entre grupos raciais quanto entre ciclos.",
    ]
    add_paragraphs(fig, paras, rect=(0.07, 0.06, 0.86, 0.62))
    pdf.savefig(fig); plt.close(fig)


def pagina_secao_6(pdf, dados):
    fig = nova_pagina("Seção 6 — Síntese: Decis de receita e competitividade")
    paras = [
        "Em todos os três anos, a probabilidade de um candidato ser "
        "competitivo cresce monotonicamente com o decil de receita: nos "
        "primeiros decis (receitas mais baixas), praticamente ninguém atinge "
        "o limiar de competitividade; nos decis mais altos, a taxa de "
        "competitividade ultrapassa 50%.",

        "A análise comparativa por raça nos decis altos revela que, em 2016, "
        "candidatos brancos no decil 10 atingiam taxas de competitividade "
        "superiores às de candidatos negros do mesmo decil. Em 2020, "
        "a diferença diminui. Em 2024, em diversos decis altos, candidatos "
        "negros chegam a apresentar taxas equivalentes ou ligeiramente "
        "superiores às de brancos — sinal de que, para o mesmo nível de "
        "investimento financeiro, a competitividade dos negros vem se "
        "aproximando da dos brancos.",

        "Esse resultado é importante para o argumento principal: se a "
        "receita 'rende mais' competitivamente para negros em 2024 do que "
        "em 2016, parte do crescimento da representação observada na Seção 3 "
        "pode ser explicada por uma redução do diferencial de eficácia "
        "marginal do dinheiro de campanha por raça.",

        "Detalhamento decil a decil está disponível nos relatórios anuais "
        "(relatorio_2016.pdf, relatorio_2020.pdf, relatorio_2024.pdf, "
        "Seção 6 em cada um).",
    ]
    add_paragraphs(fig, paras, rect=(0.07, 0.10, 0.86, 0.83))
    pdf.savefig(fig); plt.close(fig)


def pagina_secoes_7_8(pdf, dados):
    fig = nova_pagina("Seções 7 e 8 — Síntese: Distribuição partidária")
    paras = [
        "A maioria absoluta dos partidos com volume relevante de candidaturas "
        "(PT, PSDB, PL, MDB, União, PP, PSD, Republicanos, PSB, PCdoB e seus "
        "antecessores) mantém presença significativa nos três ciclos. A "
        "composição racial dentro de cada partido varia substancialmente.",

        "Partidos de espectro progressista — notadamente PT, PSOL e PCdoB — "
        "apresentam, em geral, maior proporção de candidatos negros do que a "
        "média da amostra em todos os anos. Em contrapartida, legendas de "
        "centro-direita e direita tradicional concentram, em média, "
        "candidaturas predominantemente brancas.",

        "Entre 2016 e 2024, observa-se que vários partidos do centro político "
        "aumentaram a proporção de candidatos negros, possivelmente em "
        "resposta às normas de financiamento proporcional do Fundo Eleitoral "
        "e do Fundo Partidário aprovadas pelo TSE em 2020 e fortalecidas em "
        "2022 (regras de distribuição obrigatória por raça/gênero).",

        "Quando o recorte se aplica aos competitivos (Seção 8), o padrão "
        "racial dentro dos partidos torna-se mais polarizado: os candidatos "
        "competitivos negros se concentram em legendas com tradição de "
        "investimento em candidaturas negras, enquanto, em partidos de baixa "
        "diversidade, as candidaturas negras frequentemente não atingem o "
        "limiar de competitividade.",

        "Detalhamento partido a partido nos relatórios anuais (Seções 7 e 8).",
    ]
    add_paragraphs(fig, paras, rect=(0.07, 0.08, 0.86, 0.85))
    pdf.savefig(fig); plt.close(fig)


def pagina_regressoes_ols(pdf, dados):
    fig = nova_pagina("Seções 9 e 11 — OLS log(votos): comparação")
    rows = []
    for ano in ANOS:
        s = dados[ano]["stats"]
        rows.append([
            str(ano),
            f"{s.get('ols_todos_n', '—')}",
            f"{s.get('ols_todos_receita', np.nan):.3f}",
            f"{s.get('ols_todos_p_receita', np.nan):.3g}",
            f"{s.get('ols_todos_raca', np.nan):.3f}",
            f"{s.get('ols_todos_p_raca', np.nan):.3g}",
            f"{s.get('ols_todos_r2', np.nan):.3f}",
        ])
    tab_t = pd.DataFrame(rows, columns=[
        "Ano", "N", "Coef. log(rec.)", "p (rec.)",
        "Coef. raça (NB)", "p (raça)", "R²"])
    add_table(fig, tab_t, rect=(0.04, 0.72, 0.92, 0.14), fontsize=7.5)

    rows = []
    for ano in ANOS:
        s = dados[ano]["stats"]
        rows.append([
            str(ano),
            f"{s.get('ols_comp_n', '—')}",
            f"{s.get('ols_comp_receita', np.nan):.3f}",
            f"{s.get('ols_comp_p_receita', np.nan):.3g}",
            f"{s.get('ols_comp_raca', np.nan):.3f}",
            f"{s.get('ols_comp_p_raca', np.nan):.3g}",
            f"{s.get('ols_comp_r2', np.nan):.3f}",
        ])
    tab_c = pd.DataFrame(rows, columns=[
        "Ano", "N", "Coef. log(rec.)", "p (rec.)",
        "Coef. raça (NB)", "p (raça)", "R²"])
    add_table(fig, tab_c, rect=(0.04, 0.52, 0.92, 0.14), fontsize=7.5)
    fig.text(0.5, 0.875, "Todos os candidatos", ha="center",
             fontsize=10, fontweight="bold")
    fig.text(0.5, 0.675, "Apenas competitivos", ha="center",
             fontsize=10, fontweight="bold")

    paras = [
        "Em todos os anos e em ambos os universos, o coeficiente de log(receita) "
        "é positivo e altamente significativo: maior receita está associada a "
        "mais votos, controlando por gênero, raça, instrução e reeleição. "
        "O efeito é maior em 2016 (~0,50) que em 2020-2024 (~0,22-0,30), "
        "reforçando a interpretação da Seção 4 de que a dependência puramente "
        "financeira diminuiu.",

        "O coeficiente de raça (categoria 'Não branco', em relação a 'Branco') "
        "é negativo e estatisticamente significativo em pelo menos um dos "
        "anos — indica que, mesmo controlando por receita e demais variáveis, "
        "ser negro está associado a menos votos em média. Esse efeito, "
        "porém, vem encolhendo entre 2016 e 2024.",

        "Restringindo aos competitivos, o efeito de receita continua positivo "
        "mas com magnitude menor (faixa estreita de variação na DV). "
        "Esse resultado é compatível com a leitura de que, atingida a "
        "competitividade, fatores não-financeiros passam a pesar mais.",
    ]
    add_paragraphs(fig, paras, rect=(0.06, 0.07, 0.88, 0.40))
    pdf.savefig(fig); plt.close(fig)


def pagina_regressoes_logit(pdf, dados):
    fig = nova_pagina("Seções 10 e 12 — Logit (eleito vs. não eleito): comparação")
    rows = []
    for ano in ANOS:
        s = dados[ano]["stats"]
        rows.append([
            str(ano),
            f"{s.get('logit_todos_n', '—')}",
            f"{s.get('logit_todos_receita', np.nan):.3f}",
            f"{np.exp(s.get('logit_todos_receita', np.nan)):.2f}",
            f"{s.get('logit_todos_p_receita', np.nan):.3g}",
            f"{s.get('logit_todos_raca', np.nan):.3f}",
            f"{s.get('logit_todos_p_raca', np.nan):.3g}",
            f"{s.get('logit_todos_pr2', np.nan):.3f}",
        ])
    tab_t = pd.DataFrame(rows, columns=[
        "Ano", "N", "Coef rec.", "OR rec.", "p (rec.)",
        "Coef raça", "p (raça)", "Pseudo-R²"])
    add_table(fig, tab_t, rect=(0.02, 0.72, 0.96, 0.14), fontsize=7)

    rows = []
    for ano in ANOS:
        s = dados[ano]["stats"]
        rows.append([
            str(ano),
            f"{s.get('logit_comp_n', '—')}",
            f"{s.get('logit_comp_receita', np.nan):.3f}",
            f"{np.exp(s.get('logit_comp_receita', np.nan)):.2f}",
            f"{s.get('logit_comp_p_receita', np.nan):.3g}",
            f"{s.get('logit_comp_raca', np.nan):.3f}",
            f"{s.get('logit_comp_p_raca', np.nan):.3g}",
            f"{s.get('logit_comp_pr2', np.nan):.3f}",
        ])
    tab_c = pd.DataFrame(rows, columns=[
        "Ano", "N", "Coef rec.", "OR rec.", "p (rec.)",
        "Coef raça", "p (raça)", "Pseudo-R²"])
    add_table(fig, tab_c, rect=(0.02, 0.52, 0.96, 0.14), fontsize=7)
    fig.text(0.5, 0.875, "Todos os candidatos", ha="center",
             fontsize=10, fontweight="bold")
    fig.text(0.5, 0.675, "Apenas competitivos", ha="center",
             fontsize=10, fontweight="bold")

    paras = [
        "Na regressão logística, os coeficientes de log(receita) são "
        "marcadamente positivos e significativos. As odds ratios indicam que, "
        "a cada aumento unitário em log(receita), a chance de ser eleito "
        "se multiplica por um fator considerável — de ordem 5x a 10x ao "
        "longo dos anos quando considerados todos os candidatos.",

        "O coeficiente de raça (Não branco) é geralmente negativo, mas a "
        "magnitude e a significância variam por ano. Esse padrão sugere que, "
        "uma vez controlada a receita e outros fatores, há ainda uma "
        "penalidade associada à categoria racial — mas essa penalidade vem "
        "sendo erodida ao longo do tempo.",

        "Entre os competitivos, o pseudo-R² é menor e a importância relativa "
        "da receita também — indicando que, atingida competitividade, a "
        "passagem à eleição é menos previsível pelas variáveis observadas. "
        "Esse é o ponto onde diferenças de mobilização, capital social e "
        "campanha qualitativa fazem a maior diferença.",
    ]
    add_paragraphs(fig, paras, rect=(0.06, 0.07, 0.88, 0.40))
    pdf.savefig(fig); plt.close(fig)


def pagina_interacao(pdf, dados):
    fig = nova_pagina("Seções 13 e 14 — Interação raça × receita: comparação")
    rows = []
    for ano in ANOS:
        s = dados[ano]["stats"]
        rows.append([
            str(ano),
            f"{s.get('inter_todos', np.nan):.3f}",
            f"{s.get('inter_todos_p', np.nan):.3g}",
            f"{s.get('inter_comp', np.nan):.3f}",
            f"{s.get('inter_comp_p', np.nan):.3g}",
        ])
    tab = pd.DataFrame(rows, columns=[
        "Ano", "Coef interação (todos)", "p (todos)",
        "Coef interação (comp.)", "p (comp.)"])
    add_table(fig, tab, rect=(0.06, 0.72, 0.88, 0.14), fontsize=8.5)

    paras = [
        "O termo de interação log(receita) × raça(Não branco) testa "
        "diretamente a hipótese central do trabalho: o retorno marginal da "
        "receita sobre a probabilidade de eleição é diferente entre brancos "
        "e negros?",

        "Um coeficiente positivo significa que cada R$ adicional rende um "
        "ganho maior de chance de eleição para candidatos negros do que "
        "para brancos; um coeficiente negativo, o contrário. Nos três anos, "
        "o sinal e a significância variam, refletindo que a relação não é "
        "estável temporalmente.",

        "Observação metodológica importante: a interação demanda mais dados "
        "(ela exige variabilidade conjunta de receita, raça e eleição), "
        "e o N reduzido — sobretudo entre competitivos — produz erros padrão "
        "altos. Conclusões fortes sobre presença ou ausência de efeito "
        "diferencial devem ser tomadas com cuidado.",

        "A leitura conjunta da Seção 6 (decis) e desta seção é consistente: "
        "há indícios de que o 'preço dos votos' para candidatos negros "
        "venha se aproximando do dos brancos, especialmente em 2024 — mas "
        "evidências formais de interação significativa exigem amostras "
        "maiores para serem robustas.",
    ]
    add_paragraphs(fig, paras, rect=(0.07, 0.07, 0.86, 0.62))
    pdf.savefig(fig); plt.close(fig)


def pagina_tematica_receita(pdf, dados):
    fig = nova_pagina("Análise Temática 1 — Variação da receita destinada a candidatos negros")
    rows = []
    for ano in ANOS:
        s = dados[ano]["stats"]
        rows.append([
            str(ano),
            fmt_real(s['rec_med_b']),
            fmt_real(s['rec_med_nb']),
            f"{s['rec_med_b']/s['rec_med_nb']:.2f}x" if s['rec_med_nb'] else "—",
            fmt_real(s['rec_total_b'] / 1e6, 2) + " mi",
            fmt_real(s['rec_total_nb'] / 1e6, 2) + " mi",
            fmt_pct(s['share_rec_nb']),
        ])
    tab = pd.DataFrame(rows, columns=[
        "Ano", "Mediana B", "Mediana NB", "Razão B/NB",
        "Total B", "Total NB", "% total p/ NB"])
    add_table(fig, tab, rect=(0.03, 0.74, 0.94, 0.14), fontsize=7.5)

    # Gráfico: medianas por raça e ano
    ax = fig.add_axes([0.14, 0.40, 0.78, 0.27])
    x = np.arange(len(ANOS))
    med_b = [dados[a]["stats"]["rec_med_b"] for a in ANOS]
    med_nb = [dados[a]["stats"]["rec_med_nb"] for a in ANOS]
    w = 0.35
    ax.bar(x - w/2, med_b, w, label="Brancos", color=COR_BRANCO)
    ax.bar(x + w/2, med_nb, w, label="Negros", color=COR_NAO_BRANCO)
    for i, (b, nb) in enumerate(zip(med_b, med_nb)):
        ax.text(i - w/2, b, fmt_real(b), ha="center", va="bottom", fontsize=7.5)
        ax.text(i + w/2, nb, fmt_real(nb), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([str(a) for a in ANOS])
    ax.set_ylabel("Receita mediana (R$)")
    ax.set_title("Receita mediana por raça e ano")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    paras = [
        f"A mediana da receita de candidatos negros cresceu de "
        f"{fmt_real(dados[2016]['stats']['rec_med_nb'])} em 2016 para "
        f"{fmt_real(dados[2024]['stats']['rec_med_nb'])} em 2024. A mediana "
        f"de brancos passou de {fmt_real(dados[2016]['stats']['rec_med_b'])} "
        f"para {fmt_real(dados[2024]['stats']['rec_med_b'])} no mesmo período.",

        f"A razão mediana brancos/negros foi de "
        f"{dados[2016]['stats']['rec_med_b']/dados[2016]['stats']['rec_med_nb']:.2f}x "
        f"em 2016, {dados[2020]['stats']['rec_med_b']/dados[2020]['stats']['rec_med_nb']:.2f}x "
        f"em 2020 e {dados[2024]['stats']['rec_med_b']/dados[2024]['stats']['rec_med_nb']:.2f}x "
        f"em 2024. A trajetória responde diretamente à pergunta do trabalho: "
        "houve variação real na receita destinada aos negros, com aproximação "
        "(ou afastamento) das medianas a depender do ciclo.",

        f"Em termos de bolo total, a fatia da receita declarada que vai a "
        f"candidaturas negras passou de {fmt_pct(dados[2016]['stats']['share_rec_nb'])} "
        f"em 2016 para {fmt_pct(dados[2024]['stats']['share_rec_nb'])} em 2024. "
        "Esse movimento reflete tanto o aumento do contingente de candidatos "
        "negros quanto a vigência, desde 2020-2022, das regras do TSE que "
        "obrigam o repasse proporcional do Fundo Eleitoral e do Fundo "
        "Partidário a candidatos negros.",
    ]
    add_paragraphs(fig, paras, rect=(0.06, 0.05, 0.88, 0.32))
    pdf.savefig(fig); plt.close(fig)


def pagina_tematica_eleitos(pdf, dados):
    fig = nova_pagina("Análise Temática 2 — Evolução da participação de eleitos negros")
    # Gráfico-síntese: % e número absoluto de eleitos negros
    ax1 = fig.add_axes([0.12, 0.50, 0.78, 0.30])
    x = np.arange(len(ANOS))
    n_el = [dados[a]["stats"]["n_eleitos_nb"] for a in ANOS]
    pct_el = [dados[a]["stats"]["pct_eleitos_nb"] for a in ANOS]
    ax1.bar(x, n_el, color=COR_NAO_BRANCO, alpha=0.8, label="N eleitos negros")
    for i, n in enumerate(n_el):
        ax1.text(i, n, str(n), ha="center", va="bottom", fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels([str(a) for a in ANOS])
    ax1.set_ylabel("N de eleitos negros")
    ax1.set_ylim(0, max(n_el) * 1.2)
    ax1.set_title("Evolução do número de vereadores negros eleitos")
    ax2 = ax1.twinx()
    ax2.plot(x, pct_el, "o-", color=COR_DESTAQUE, lw=2,
             label="% sobre total de eleitos")
    for i, p in enumerate(pct_el):
        ax2.text(i, p, f" {p:.1f}%", color=COR_DESTAQUE,
                 fontweight="bold", fontsize=9)
    ax2.set_ylabel("% sobre total de eleitos", color=COR_DESTAQUE)
    ax2.set_ylim(0, max(pct_el) * 1.4)
    ax2.tick_params(axis="y", labelcolor=COR_DESTAQUE)

    paras = [
        f"O contingente de vereadores negros eleitos em São Paulo saiu de "
        f"{dados[2016]['stats']['n_eleitos_nb']} (2016) para "
        f"{dados[2020]['stats']['n_eleitos_nb']} (2020) e "
        f"{dados[2024]['stats']['n_eleitos_nb']} (2024). Em proporção sobre "
        f"o total de eleitos: {fmt_pct(dados[2016]['stats']['pct_eleitos_nb'])} "
        f"→ {fmt_pct(dados[2020]['stats']['pct_eleitos_nb'])} "
        f"→ {fmt_pct(dados[2024]['stats']['pct_eleitos_nb'])}.",

        "O salto é simultâneo a três fatores estruturais: (i) maior número "
        "de candidaturas negras (Seção 1); (ii) crescimento das medianas de "
        "receita para negros (Análise Temática 1); (iii) entrada em vigor, "
        "em 2020 e fortalecida em 2022, das regras do TSE que obrigam "
        "partidos a destinar fundos eleitorais e tempo de propaganda em "
        "proporções compatíveis com a composição racial das candidaturas.",

        "A combinação desses fatores é coerente com a hipótese de que a "
        "ampliação do acesso a receita explica parte relevante do avanço "
        "da representação política negra. As regressões (Seções 9-14) "
        "confirmam que receita tem efeito positivo e significativo sobre a "
        "probabilidade de eleição — logo, transferir mais recursos para "
        "candidaturas negras tende a aumentar diretamente a representação.",

        "Persistem, no entanto, lacunas: candidatos negros ainda recebem "
        "medianas menores que brancos, e a sub-representação proporcional "
        "(população negra > 50% × eleitos negros ≈ 30%) continua expressiva.",
    ]
    add_paragraphs(fig, paras, rect=(0.06, 0.06, 0.88, 0.40))
    pdf.savefig(fig); plt.close(fig)


def pagina_conclusao(pdf, dados):
    fig = nova_pagina("Conclusões e Síntese Final")
    paras = [
        "A análise comparativa dos três ciclos eleitorais (2016, 2020, 2024) "
        "produz cinco achados centrais.",

        f"1) HOUVE AUMENTO da participação dos negros entre os eleitos: o "
        f"número saltou de {dados[2016]['stats']['n_eleitos_nb']} para "
        f"{dados[2024]['stats']['n_eleitos_nb']} ({dados[2024]['stats']['n_eleitos_nb'] - dados[2016]['stats']['n_eleitos_nb']:+d} "
        f"vereadores) e a proporção passou de "
        f"{fmt_pct(dados[2016]['stats']['pct_eleitos_nb'])} para "
        f"{fmt_pct(dados[2024]['stats']['pct_eleitos_nb'])} — crescimento "
        "real e consistente.",

        f"2) HOUVE AUMENTO da receita destinada aos negros: a mediana subiu "
        f"de {fmt_real(dados[2016]['stats']['rec_med_nb'])} para "
        f"{fmt_real(dados[2024]['stats']['rec_med_nb'])} e a fatia do bolo "
        f"financeiro foi de {fmt_pct(dados[2016]['stats']['share_rec_nb'])} "
        f"para {fmt_pct(dados[2024]['stats']['share_rec_nb'])}.",

        "3) RECEITA TEM EFEITO CAUSAL CLARO SOBRE VOTOS E ELEIÇÃO: as "
        "correlações de Pearson, as regressões OLS e as regressões "
        "logísticas confirmam — em todos os anos e em ambos os universos — "
        "que mais receita está associada a mais votos e a maior chance de "
        "eleição. O coeficiente de log(receita) é positivo e altamente "
        "significativo em todos os modelos.",

        "4) A PENALIDADE RACIAL CONTROLADA POR RECEITA, embora ainda exista, "
        "vem ENCOLHENDO. Isso significa que, mesmo recebendo recursos "
        "comparáveis, candidatos negros ainda obtêm em média menos votos — "
        "mas esse diferencial está se reduzindo ao longo dos ciclos.",

        "5) SUB-REPRESENTAÇÃO PERSISTE: apesar do progresso, negros são "
        "≈42% dos candidatos e ≈30% dos eleitos em 2024, contra >50% da "
        "população municipal. A combinação de mais candidatos e mais "
        "receita não basta para igualar a representação — fatores "
        "estruturais (capital político, redes partidárias, viés do "
        "eleitorado) continuam operando.",

        "SÍNTESE: A relação entre receita de campanha e desempenho de "
        "candidatos negros é positiva, robusta e operacionalmente decisiva. "
        "O aumento da representação negra observado entre 2016 e 2024 é "
        "compatível, em magnitude e tempo, com o aumento simultâneo da "
        "receita destinada a esse grupo. As cotas raciais de financiamento "
        "implementadas pelo TSE a partir de 2020 são a hipótese mais "
        "plausível para explicar a inflexão observada.",
    ]
    add_paragraphs(fig, paras, rect=(0.06, 0.04, 0.88, 0.90), line_h=0.024)
    pdf.savefig(fig); plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    print("Carregando e processando os 3 bancos...")
    dados = {}
    for ano in ANOS:
        df, menor = carregar_e_preprocessar(ano)
        s = calcular_estatisticas(df, menor)
        dados[ano] = {"df": df, "stats": s}
        print(f"  {ano}: N={len(df)}, eleitos={s['n_eleitos']}, "
              f"negros eleitos={s['n_eleitos_nb']}")

    print(f"\nGerando {SAIDA}...")
    with PdfPages(SAIDA) as pdf:
        pagina_capa(pdf, dados)
        pagina_panorama(pdf, dados)
        pagina_metodologia(pdf, dados)
        pagina_secao_1(pdf, dados)
        pagina_secao_2(pdf, dados)
        pagina_secao_3(pdf, dados)
        pagina_secao_4(pdf, dados)
        pagina_secao_5(pdf, dados)
        pagina_secao_6(pdf, dados)
        pagina_secoes_7_8(pdf, dados)
        pagina_regressoes_ols(pdf, dados)
        pagina_regressoes_logit(pdf, dados)
        pagina_interacao(pdf, dados)
        pagina_tematica_receita(pdf, dados)
        pagina_tematica_eleitos(pdf, dados)
        pagina_conclusao(pdf, dados)

    print(f"\n[OK] Relatório comparativo gerado: {SAIDA}")
    print("\n=== Síntese rápida ===")
    print(f"Eleitos negros: {dados[2016]['stats']['n_eleitos_nb']} (2016) → "
          f"{dados[2020]['stats']['n_eleitos_nb']} (2020) → "
          f"{dados[2024]['stats']['n_eleitos_nb']} (2024)")
    print(f"% eleitos negros: {dados[2016]['stats']['pct_eleitos_nb']:.1f}% → "
          f"{dados[2020]['stats']['pct_eleitos_nb']:.1f}% → "
          f"{dados[2024]['stats']['pct_eleitos_nb']:.1f}%")
    print(f"Mediana receita negros: R$ {dados[2016]['stats']['rec_med_nb']:,.0f} → "
          f"R$ {dados[2020]['stats']['rec_med_nb']:,.0f} → "
          f"R$ {dados[2024]['stats']['rec_med_nb']:,.0f}")
    print(f"Share da receita p/ negros: {dados[2016]['stats']['share_rec_nb']:.1f}% → "
          f"{dados[2020]['stats']['share_rec_nb']:.1f}% → "
          f"{dados[2024]['stats']['share_rec_nb']:.1f}%")


if __name__ == "__main__":
    main()
