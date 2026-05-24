"""
analise_2024.py — Análise estatística de candidatos a vereador SP 2024 (TRE-SP).

Gera relatorio_2024.pdf com as análises da relação entre raça, receita de
campanha e desempenho eleitoral. Script autocontido (sem imports cruzados).
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
ANO = 2024
ARQUIVO = "Banco_2024.csv"
SAIDA = f"relatorio_{ANO}.pdf"
COR_BRANCO = "#4C72B0"
COR_NAO_BRANCO = "#DD8452"
COR_NEUTRA = "#666666"
COR_SIG = "#2ca02c"
FIG_A4 = (8.27, 11.69)
RODAPE = f"Fonte: TRE-SP, {ANO}"

np.random.seed(42)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})


# ============================================================
# Carregamento e pré-processamento
# ============================================================
def carregar():
    if not Path(ARQUIVO).exists():
        sys.exit(f"ERRO: arquivo '{ARQUIVO}' não encontrado no diretório atual.")
    df = pd.read_csv(ARQUIVO, sep=";", encoding="latin1", decimal=",")
    print(f"[carregar] {len(df)} linhas, {len(df.columns)} colunas")
    return df


def preprocessar(df):
    log = {"n_inicial": len(df), "dist_raca_original": df["DS_COR_RACA"].value_counts().to_dict()}

    df = df.dropna(subset=["TOTAL_VOTOS", "DS_SIT_TOT_TURNO"]).copy()
    df["TOTAL_VOTOS"] = df["TOTAL_VOTOS"].astype(int)
    log["n_apos_dropna"] = len(df)

    df["ELEITO"] = df["DS_SIT_TOT_TURNO"].isin(["ELEITO POR QP", "ELEITO POR MÉDIA"]).astype(int)
    log["n_eleitos"] = int(df["ELEITO"].sum())

    menor_eleito = int(df.loc[df["ELEITO"] == 1, "TOTAL_VOTOS"].min())
    limiar = 0.20 * menor_eleito
    df["COMPETITIVO"] = (df["TOTAL_VOTOS"] >= limiar).astype(int)
    log["menor_eleito_votos"] = menor_eleito
    log["limiar_competitivo"] = limiar
    log["n_competitivos"] = int(df["COMPETITIVO"].sum())

    raca_map = {"BRANCA": "Branco", "PRETA": "Não branco", "PARDA": "Não branco"}
    df["RACA_BIN"] = df["DS_COR_RACA"].map(raca_map)
    log["n_excluidos_raca"] = int(df["RACA_BIN"].isna().sum())

    df = df[df["RACA_BIN"].notna()].copy()
    log["n_apos_filtro_raca"] = len(df)
    log["n_branco"] = int((df["RACA_BIN"] == "Branco").sum())
    log["n_nao_branco"] = int((df["RACA_BIN"] == "Não branco").sum())

    df["LOG_RECEITA"] = np.log1p(df["RECEITA_TOTAL"])
    df["LOG_VOTOS"] = np.log1p(df["TOTAL_VOTOS"])
    df["LOG_BENS"] = np.log1p(df["BENS_TOTAL"])

    return df, log


# ============================================================
# Helpers de página
# ============================================================
def nova_pagina(titulo):
    fig = plt.figure(figsize=FIG_A4)
    fig.suptitle(titulo, fontsize=13, fontweight="bold", y=0.965)
    fig.text(0.5, 0.015, RODAPE, ha="center", fontsize=8, color=COR_NEUTRA)
    return fig


def add_table(fig, df_table, rect, fontsize=8):
    ax = fig.add_axes(rect)
    ax.axis("off")
    tab = ax.table(
        cellText=df_table.values.tolist(),
        colLabels=df_table.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(fontsize)
    tab.scale(1, 1.5)
    for j in range(len(df_table.columns)):
        tab[(0, j)].set_text_props(fontweight="bold", color="white")
        tab[(0, j)].set_facecolor("#444")
    return ax


def add_text(fig, texto, rect, fontsize=9):
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.text(0, 1, texto, fontsize=fontsize, va="top", ha="left", wrap=True)
    return ax


def add_chart(fig, rect):
    return fig.add_axes(rect)


# ============================================================
# Páginas de capa e caracterização
# ============================================================
def pagina_capa(pdf, log):
    fig = plt.figure(figsize=FIG_A4)
    fig.text(0.5, 0.78, "Relatório de Análise Estatística", ha="center",
             fontsize=22, fontweight="bold")
    fig.text(0.5, 0.71, f"Candidatos a Vereador — São Paulo {ANO}",
             ha="center", fontsize=18)
    fig.text(0.5, 0.62, "Raça, Receita de Campanha e Desempenho Eleitoral",
             ha="center", fontsize=13, fontstyle="italic")

    info = (
        f"N total inicial: {log['n_inicial']:,}\n"
        f"N após filtros (apenas Brancos e Negros): {log['n_apos_filtro_raca']:,}\n"
        f"N eleitos: {log['n_eleitos']}\n"
        f"N competitivos: {log['n_competitivos']}\n"
        f"Limiar de competitividade: {log['limiar_competitivo']:.0f} votos\n"
        f"(20% do menor eleito = {log['menor_eleito_votos']:,} votos)"
    )
    fig.text(0.5, 0.40, info, ha="center", fontsize=11,
             bbox=dict(boxstyle="round,pad=1", fc="#f0f0f0", ec="#bbb"))
    fig.text(0.5, 0.15, f"Gerado em {datetime.now():%d/%m/%Y %H:%M}",
             ha="center", fontsize=9, color=COR_NEUTRA)
    pdf.savefig(fig); plt.close(fig)


def pagina_caracterizacao(pdf, df, log):
    fig = nova_pagina("Seção 0 — Caracterização da Amostra")

    tab = pd.DataFrame({
        "Métrica": [
            "N total na base",
            "N após remoção de NA em votos/situação",
            "N excluídos por raça (fora de Branca/Preta/Parda)",
            "N final (amostra analítica)",
            "N eleitos",
            "Menor número de votos entre eleitos",
            "Limiar de competitividade (20%)",
            "N candidatos competitivos",
            "N brancos",
            "N não brancos",
            "NA em RECEITA_TOTAL",
            "NA em BENS_TOTAL",
        ],
        "Valor": [
            f"{log['n_inicial']:,}",
            f"{log['n_apos_dropna']:,}",
            f"{log['n_excluidos_raca']:,}",
            f"{log['n_apos_filtro_raca']:,}",
            f"{log['n_eleitos']:,}",
            f"{log['menor_eleito_votos']:,}",
            f"{log['limiar_competitivo']:.0f}",
            f"{log['n_competitivos']:,}",
            f"{log['n_branco']:,}",
            f"{log['n_nao_branco']:,}",
            f"{df['RECEITA_TOTAL'].isna().sum():,}",
            f"{df['BENS_TOTAL'].isna().sum():,}",
        ],
    })
    add_table(fig, tab, rect=(0.10, 0.43, 0.80, 0.47), fontsize=9)

    dist = pd.Series(log["dist_raca_original"]).sort_values(ascending=False)
    ax = add_chart(fig, rect=(0.14, 0.08, 0.76, 0.28))
    bars = ax.barh(dist.index.astype(str), dist.values, color=COR_NEUTRA)
    ax.set_title("Distribuição original por cor/raça (antes do recorte binário)")
    ax.invert_yaxis()
    ax.set_xlabel("Contagem")
    for bar, v in zip(bars, dist.values):
        ax.text(v, bar.get_y() + bar.get_height() / 2, f" {v}",
                va="center", fontsize=8)
    pdf.savefig(fig); plt.close(fig)


# ============================================================
# Seções 1-3: Proporções
# ============================================================
def pagina_proporcao(pdf, universo, titulo, secao):
    fig = nova_pagina(f"Seção {secao} — {titulo}")
    counts = universo["RACA_BIN"].value_counts()
    n_b = int(counts.get("Branco", 0))
    n_nb = int(counts.get("Não branco", 0))
    total = n_b + n_nb

    tab = pd.DataFrame({
        "Raça": ["Branco", "Não branco", "Total"],
        "N": [n_b, n_nb, total],
        "Proporção": [
            f"{n_b/total*100:.1f}%" if total else "—",
            f"{n_nb/total*100:.1f}%" if total else "—",
            "100,0%" if total else "—",
        ],
    })
    add_table(fig, tab, rect=(0.25, 0.74, 0.50, 0.16))

    ax = add_chart(fig, rect=(0.22, 0.32, 0.56, 0.36))
    labels = ["Branco", "Não branco"]
    valores = [n_b, n_nb]
    bars = ax.bar(labels, valores, color=[COR_BRANCO, COR_NAO_BRANCO])
    ax.set_ylabel("Número de candidatos")
    for bar, v in zip(bars, valores):
        pct = v / total * 100 if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f"{v}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(valores) * 1.18 if max(valores) else 1)

    pct_b = n_b / total * 100 if total else 0
    pct_nb = n_nb / total * 100 if total else 0
    diff = pct_b - pct_nb
    interp = (
        f"Universo analisado: {total} candidatos.\n"
        f"• Brancos: {n_b} ({pct_b:.1f}%)\n"
        f"• Não brancos (pretos + pardos): {n_nb} ({pct_nb:.1f}%)\n"
        f"Diferença: {abs(diff):.1f} p.p. a favor dos "
        f"{'brancos' if diff > 0 else 'não brancos'}."
    )
    add_text(fig, interp, rect=(0.10, 0.05, 0.80, 0.22))
    pdf.savefig(fig); plt.close(fig)


# ============================================================
# Seções 4-5: Correlação Pearson
# ============================================================
def correlacao_por_raca(df):
    res = {}
    for raca in ["Branco", "Não branco"]:
        sub = df[(df["RACA_BIN"] == raca)
                 & df["RECEITA_TOTAL"].notna()
                 & df["TOTAL_VOTOS"].notna()]
        if len(sub) >= 3:
            r, p = stats.pearsonr(sub["RECEITA_TOTAL"], sub["TOTAL_VOTOS"])
            res[raca] = {"n": len(sub), "r": r, "p": p}
        else:
            res[raca] = {"n": len(sub), "r": np.nan, "p": np.nan}
    return res


def pagina_correlacao(pdf, universo, titulo, secao):
    fig = nova_pagina(f"Seção {secao} — {titulo}")
    res = correlacao_por_raca(universo)

    tab = pd.DataFrame({
        "Raça": ["Branco", "Não branco"],
        "N": [res["Branco"]["n"], res["Não branco"]["n"]],
        "r (Pearson)": [f"{res['Branco']['r']:.4f}", f"{res['Não branco']['r']:.4f}"],
        "p-valor": [f"{res['Branco']['p']:.4g}", f"{res['Não branco']['p']:.4g}"],
    })
    add_table(fig, tab, rect=(0.18, 0.76, 0.64, 0.14))

    ax = add_chart(fig, rect=(0.14, 0.30, 0.78, 0.40))
    for raca, cor in [("Branco", COR_BRANCO), ("Não branco", COR_NAO_BRANCO)]:
        sub = universo[(universo["RACA_BIN"] == raca)
                       & universo["RECEITA_TOTAL"].notna()
                       & universo["TOTAL_VOTOS"].notna()]
        ax.scatter(sub["RECEITA_TOTAL"], sub["TOTAL_VOTOS"],
                   color=cor, alpha=0.45, s=18, label=raca,
                   edgecolor="none")
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.set_xlabel("Receita total da campanha (R$, escala log)")
    ax.set_ylabel("Número de votos (escala log)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    r_b = res["Branco"]["r"]; r_nb = res["Não branco"]["r"]
    mais_forte = "brancos" if (pd.notna(r_b) and pd.notna(r_nb) and r_b > r_nb) else "não brancos"
    interp = (
        f"Correlação de Pearson entre receita e votos por raça:\n"
        f"  • Brancos: r = {r_b:.3f} (p = {res['Branco']['p']:.3g}, n = {res['Branco']['n']})\n"
        f"  • Não brancos: r = {r_nb:.3f} (p = {res['Não branco']['p']:.3g}, n = {res['Não branco']['n']})\n"
        f"A correlação é mais forte entre {mais_forte}, indicando que cada R$ a mais "
        f"de receita tem uma associação relativamente mais intensa com votos nesse grupo."
    )
    add_text(fig, interp, rect=(0.08, 0.04, 0.84, 0.22))
    pdf.savefig(fig); plt.close(fig)


# ============================================================
# Seção 6: Decis de receita
# ============================================================
def pagina_decis_receita(pdf, df, secao):
    fig = nova_pagina(f"Seção {secao} — Proporção de competitivos brancos e não brancos por decil de receita")

    sub = df[df["RECEITA_TOTAL"].notna() & (df["RECEITA_TOTAL"] > 0)].copy()
    sub["decil"] = pd.qcut(sub["RECEITA_TOTAL"], 10, labels=range(1, 11), duplicates="drop")

    grouped = (sub.groupby(["decil", "RACA_BIN"], observed=True)
               .agg(n=("COMPETITIVO", "size"), n_comp=("COMPETITIVO", "sum"))
               .reset_index())
    grouped["pct_comp"] = grouped["n_comp"] / grouped["n"] * 100

    pct_pivot = grouped.pivot(index="decil", columns="RACA_BIN", values="pct_comp")
    n_pivot = grouped.pivot(index="decil", columns="RACA_BIN", values="n")

    ax = add_chart(fig, rect=(0.12, 0.30, 0.80, 0.45))
    x = np.arange(len(pct_pivot))
    width = 0.38
    if "Branco" in pct_pivot.columns:
        ax.bar(x - width / 2, pct_pivot["Branco"].fillna(0), width,
               label="Branco", color=COR_BRANCO)
    if "Não branco" in pct_pivot.columns:
        ax.bar(x + width / 2, pct_pivot["Não branco"].fillna(0), width,
               label="Não branco", color=COR_NAO_BRANCO)
    ax.set_xticks(x)
    ax.set_xticklabels(pct_pivot.index.astype(str))
    ax.set_xlabel("Decil de receita (1 = menor, 10 = maior)")
    ax.set_ylabel("% candidatos competitivos no decil")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    tab = pct_pivot.reset_index().copy()
    tab["decil"] = tab["decil"].astype(str)
    for col in tab.columns[1:]:
        tab[col] = tab[col].apply(lambda v: f"{v:.1f}%" if pd.notna(v) else "—")
    add_table(fig, tab, rect=(0.22, 0.78, 0.56, 0.12), fontsize=7)

    try:
        b_top = pct_pivot["Branco"].iloc[-1]; b_bot = pct_pivot["Branco"].iloc[0]
        nb_top = pct_pivot["Não branco"].iloc[-1]; nb_bot = pct_pivot["Não branco"].iloc[0]
        interp = (
            f"Entre o 1º e o 10º decil de receita, a taxa de competitividade dos brancos "
            f"varia de {b_bot:.1f}% para {b_top:.1f}% e a dos não brancos varia de "
            f"{nb_bot:.1f}% para {nb_top:.1f}%. Quanto maior a receita, maior a chance "
            f"de o candidato ser competitivo — mas o ganho marginal entre os grupos "
            f"pode diferir, evidência preliminar de efeito diferencial da receita por raça."
        )
    except Exception:
        interp = "Decis com poucos dados em algum grupo racial; ver tabela acima."
    add_text(fig, interp, rect=(0.08, 0.04, 0.84, 0.18))
    pdf.savefig(fig); plt.close(fig)


# ============================================================
# Seções 7-8: Partidos
# ============================================================
def pagina_partidos(pdf, universo, titulo, secao, top=15):
    fig = nova_pagina(f"Seção {secao} — {titulo}")
    contagem = universo.groupby(["SG_PARTIDO", "RACA_BIN"]).size().unstack(fill_value=0)
    contagem["total"] = contagem.sum(axis=1)
    contagem = contagem.sort_values("total", ascending=False)

    if len(contagem) > top:
        top_df = contagem.head(top).copy()
        outros = contagem.iloc[top:].drop(columns="total", errors="ignore").sum(axis=0)
        outros.name = "Outros"
        top_df = pd.concat([top_df.drop(columns="total"),
                            pd.DataFrame([outros])])
    else:
        top_df = contagem.drop(columns="total")

    top_df = top_df.iloc[::-1]  # ordem ascendente para barh

    ax = add_chart(fig, rect=(0.20, 0.10, 0.72, 0.74))
    y = np.arange(len(top_df))
    if "Branco" in top_df.columns:
        ax.barh(y - 0.20, top_df["Branco"], height=0.4,
                label="Branco", color=COR_BRANCO)
    if "Não branco" in top_df.columns:
        ax.barh(y + 0.20, top_df["Não branco"], height=0.4,
                label="Não branco", color=COR_NAO_BRANCO)
    ax.set_yticks(y)
    ax.set_yticklabels(top_df.index)
    ax.set_xlabel("Número de candidatos")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    interp = (
        f"Top {min(top, len(contagem))} partidos por volume (demais agrupados em 'Outros'). "
        f"Comparação visual da composição racial em cada legenda."
    )
    add_text(fig, interp, rect=(0.08, 0.03, 0.84, 0.05))
    pdf.savefig(fig); plt.close(fig)


# ============================================================
# Seções 9-14: Regressões
# ============================================================
def df_para_modelo(df, cols):
    return df.dropna(subset=cols).copy()


def ajustar_ols(df):
    cols = ["LOG_VOTOS", "LOG_RECEITA", "DS_GENERO", "RACA_BIN",
            "DS_GRAU_INSTRUCAO", "ST_REELEICAO"]
    sub = df_para_modelo(df, cols)
    formula = ("LOG_VOTOS ~ LOG_RECEITA + C(DS_GENERO) + C(RACA_BIN) "
               "+ C(DS_GRAU_INSTRUCAO) + C(ST_REELEICAO)")
    return smf.ols(formula, data=sub).fit(), sub


def ajustar_logit(df):
    cols = ["ELEITO", "LOG_RECEITA", "DS_GENERO", "RACA_BIN", "ST_REELEICAO"]
    sub = df_para_modelo(df, cols)
    formula = "ELEITO ~ LOG_RECEITA + C(DS_GENERO) + C(RACA_BIN) + C(ST_REELEICAO)"
    return smf.logit(formula, data=sub).fit(disp=False), sub


def ajustar_logit_interacao(df):
    cols = ["ELEITO", "LOG_RECEITA", "DS_GENERO", "RACA_BIN", "ST_REELEICAO"]
    sub = df_para_modelo(df, cols)
    formula = ("ELEITO ~ LOG_RECEITA * C(RACA_BIN) "
               "+ C(DS_GENERO) + C(ST_REELEICAO)")
    return smf.logit(formula, data=sub).fit(disp=False), sub


def regressao_table(res):
    ci = res.conf_int()
    return pd.DataFrame({
        "Variável": [_abrev(v) for v in res.params.index],
        "Coef.": [f"{v:.4f}" for v in res.params.values],
        "EP": [f"{v:.4f}" for v in res.bse.values],
        "p-valor": [f"{v:.4g}" for v in res.pvalues.values],
        "IC 95%": [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(ci[0], ci[1])],
    })


def _abrev(nome):
    nome = nome.replace("C(DS_GRAU_INSTRUCAO)", "GRAU_INSTR")
    nome = nome.replace("C(DS_GENERO)", "GENERO")
    nome = nome.replace("C(RACA_BIN)", "RACA")
    nome = nome.replace("C(ST_REELEICAO)", "REELEICAO")
    nome = nome.replace("[T.", "[")
    return nome


def pagina_regressao(pdf, res, sub, tipo, titulo, secao):
    fig = nova_pagina(f"Seção {secao} — {titulo}")
    tab = regressao_table(res)
    add_table(fig, tab, rect=(0.05, 0.50, 0.90, 0.36), fontsize=7)

    if tipo == "ols":
        metricas = (f"OLS  |  R² = {res.rsquared:.3f}  |  R² aj. = {res.rsquared_adj:.3f}"
                    f"  |  N = {int(res.nobs)}  |  F = {res.fvalue:.2f}  "
                    f"(p = {res.f_pvalue:.3g})")
    else:
        metricas = (f"Logit  |  Pseudo-R² = {res.prsquared:.3f}  |  LL = {res.llf:.1f}"
                    f"  |  AIC = {res.aic:.1f}  |  N = {int(res.nobs)}")
    fig.text(0.5, 0.46, metricas, ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", ec="#bbb"))

    ax = add_chart(fig, rect=(0.20, 0.10, 0.72, 0.32))
    params = res.params.drop("Intercept", errors="ignore")
    ci = res.conf_int().drop("Intercept", errors="ignore")
    pvals = res.pvalues.drop("Intercept", errors="ignore")
    ordem = list(reversed(params.index.tolist()))
    yvals = np.arange(len(ordem))
    for i, var in enumerate(ordem):
        lo, hi = ci.loc[var, 0], ci.loc[var, 1]
        cor = COR_SIG if pvals[var] < 0.05 else COR_NEUTRA
        ax.plot([lo, hi], [yvals[i]] * 2, color=cor, lw=1.7)
        ax.plot(params[var], yvals[i], "o", color=cor, ms=6)
    ax.axvline(0, color="black", lw=0.6, ls="--", alpha=0.6)
    ax.set_yticks(yvals)
    ax.set_yticklabels([_abrev(v) for v in ordem], fontsize=7)
    ax.set_xlabel("Coeficiente (verde = p < 0.05)")
    ax.set_title("Coeficientes com IC 95%")
    ax.grid(axis="x", alpha=0.3)
    pdf.savefig(fig); plt.close(fig)


# ============================================================
# Sumário comparativo
# ============================================================
def pagina_sumario(pdf, sumario):
    fig = nova_pagina("Sumário Comparativo")
    df_sum = pd.DataFrame(sumario)
    add_table(fig, df_sum, rect=(0.05, 0.30, 0.90, 0.60), fontsize=7)
    interp = (
        "Tabela consolida principais coeficientes e correlações observados. "
        "Significância em destaque (p < 0.05). Use como leitura rápida das "
        "diferenças entre grupos raciais e entre modelos."
    )
    add_text(fig, interp, rect=(0.08, 0.18, 0.84, 0.10))
    pdf.savefig(fig); plt.close(fig)


def coletar_sumario(df, competitivos, eleitos, correl_todos, correl_comp,
                    ols_todos, logit_todos, ols_comp, logit_comp,
                    interacao_todos, interacao_comp):
    items, ests, vals = [], [], []

    def add(item, est, val):
        items.append(item); ests.append(est); vals.append(val)

    add("Brancos no total", "N (%)",
        f"{(df['RACA_BIN']=='Branco').sum()} ({(df['RACA_BIN']=='Branco').mean()*100:.1f}%)")
    add("Não brancos no total", "N (%)",
        f"{(df['RACA_BIN']=='Não branco').sum()} ({(df['RACA_BIN']=='Não branco').mean()*100:.1f}%)")
    add("Brancos eleitos", "N (% dos eleitos)",
        f"{(eleitos['RACA_BIN']=='Branco').sum()} ({(eleitos['RACA_BIN']=='Branco').mean()*100:.1f}%)")
    add("Não brancos eleitos", "N (% dos eleitos)",
        f"{(eleitos['RACA_BIN']=='Não branco').sum()} ({(eleitos['RACA_BIN']=='Não branco').mean()*100:.1f}%)")
    add("Correlação r — todos (Brancos)", "Pearson", f"{correl_todos['Branco']['r']:.3f}")
    add("Correlação r — todos (Não brancos)", "Pearson", f"{correl_todos['Não branco']['r']:.3f}")
    add("Correlação r — competitivos (Brancos)", "Pearson", f"{correl_comp['Branco']['r']:.3f}")
    add("Correlação r — competitivos (Não brancos)", "Pearson", f"{correl_comp['Não branco']['r']:.3f}")

    for nome, mod in [("OLS todos", ols_todos), ("OLS competitivos", ols_comp)]:
        if mod is not None:
            res = mod
            if "LOG_RECEITA" in res.params.index:
                add(f"{nome}: efeito log(receita)", "Coef. (p)",
                    f"{res.params['LOG_RECEITA']:.3f} (p={res.pvalues['LOG_RECEITA']:.3g})")

    for nome, mod in [("Logit todos", logit_todos), ("Logit competitivos", logit_comp)]:
        if mod is not None:
            res = mod
            if "LOG_RECEITA" in res.params.index:
                add(f"{nome}: efeito log(receita) [odds]", "exp(coef)",
                    f"{np.exp(res.params['LOG_RECEITA']):.3f} (p={res.pvalues['LOG_RECEITA']:.3g})")

    for nome, mod in [("Interação todos", interacao_todos),
                       ("Interação competitivos", interacao_comp)]:
        if mod is not None:
            chave = "LOG_RECEITA:C(RACA_BIN)[T.Não branco]"
            if chave in mod.params.index:
                add(f"{nome}: efeito interação raça×receita", "Coef. (p)",
                    f"{mod.params[chave]:.3f} (p={mod.pvalues[chave]:.3g})")

    return {"Item": items, "Estatística": ests, "Valor": vals}


# ============================================================
# Main
# ============================================================
def main():
    df_raw = carregar()
    df, log = preprocessar(df_raw)
    competitivos = df[df["COMPETITIVO"] == 1].copy()
    eleitos = df[df["ELEITO"] == 1].copy()

    ols_todos = logit_todos = ols_comp = logit_comp = None
    inter_todos = inter_comp = None

    with PdfPages(SAIDA) as pdf:
        pagina_capa(pdf, log)
        pagina_caracterizacao(pdf, df, log)

        pagina_proporcao(pdf, df,
                         "Proporção de candidatos brancos e não brancos", "1")
        pagina_proporcao(pdf, competitivos,
                         "Proporção de candidatos competitivos brancos e não brancos", "2")
        pagina_proporcao(pdf, eleitos,
                         "Proporção de candidatos eleitos brancos e não brancos", "3")

        pagina_correlacao(pdf, df,
                          "Correlação de Pearson receita × votos por raça (todos)", "4")
        pagina_correlacao(pdf, competitivos,
                          "Correlação de Pearson receita × votos por raça (competitivos)", "5")

        pagina_decis_receita(pdf, df, "6")

        pagina_partidos(pdf, df,
                        "Candidatos brancos e não brancos por partido (todos)", "7")
        pagina_partidos(pdf, competitivos,
                        "Candidatos competitivos brancos e não brancos por partido", "8")

        for label, fn, universo, secao, tipo in [
            ("OLS todos", ajustar_ols, df, "9", "ols"),
            ("Logit todos", ajustar_logit, df, "10", "logit"),
            ("OLS competitivos", ajustar_ols, competitivos, "11", "ols"),
            ("Logit competitivos", ajustar_logit, competitivos, "12", "logit"),
            ("Logit interação todos", ajustar_logit_interacao, df, "13", "logit"),
            ("Logit interação competitivos", ajustar_logit_interacao, competitivos, "14", "logit"),
        ]:
            try:
                res, sub = fn(universo)
                titulo_map = {
                    "9": "Regressão múltipla (OLS) — todos os candidatos",
                    "10": "Regressão logística — todos os candidatos",
                    "11": "Regressão múltipla (OLS) — candidatos competitivos",
                    "12": "Regressão logística — candidatos competitivos",
                    "13": "Logística com interação raça × receita — todos",
                    "14": "Logística com interação raça × receita — competitivos",
                }
                pagina_regressao(pdf, res, sub, tipo, titulo_map[secao], secao)
                if secao == "9":  ols_todos = res
                if secao == "10": logit_todos = res
                if secao == "11": ols_comp = res
                if secao == "12": logit_comp = res
                if secao == "13": inter_todos = res
                if secao == "14": inter_comp = res
            except Exception as e:
                print(f"[ERRO] {label}: {e}")

        correl_todos = correlacao_por_raca(df)
        correl_comp = correlacao_por_raca(competitivos)
        sumario = coletar_sumario(df, competitivos, eleitos,
                                  correl_todos, correl_comp,
                                  ols_todos, logit_todos, ols_comp, logit_comp,
                                  inter_todos, inter_comp)
        pagina_sumario(pdf, sumario)

    print(f"\n[OK] Relatório gerado: {SAIDA}")
    print("\n=== Resumo executivo ===")
    print(f"N total analítico: {log['n_apos_filtro_raca']}")
    print(f"N brancos / não brancos: {log['n_branco']} / {log['n_nao_branco']}")
    print(f"N eleitos: {log['n_eleitos']} "
          f"(Brancos: {(eleitos['RACA_BIN']=='Branco').sum()}, "
          f"Não brancos: {(eleitos['RACA_BIN']=='Não branco').sum()})")
    print(f"N competitivos: {log['n_competitivos']} (limiar: {log['limiar_competitivo']:.0f} votos)")
    if ols_todos is not None:
        print(f"OLS todos: R² = {ols_todos.rsquared:.3f}, "
              f"log(receita) coef = {ols_todos.params.get('LOG_RECEITA', np.nan):.3f}")
    if logit_todos is not None:
        print(f"Logit todos: pseudo-R² = {logit_todos.prsquared:.3f}, "
              f"log(receita) coef = {logit_todos.params.get('LOG_RECEITA', np.nan):.3f}")


if __name__ == "__main__":
    main()
