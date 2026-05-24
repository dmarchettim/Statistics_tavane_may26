# CLAUDE.md — Projeto Estatística Tavané

## Visão geral

Trabalho acadêmico de estatística analisando candidatos a vereador em São Paulo nas eleições de **2016**, **2020** e **2024** (dados do TRE-SP). Foco: relação entre **raça**, **receita de campanha** e **desempenho eleitoral**.

Entregável final: um **PDF de relatório por ano** (`relatorio_2016.pdf`, `relatorio_2020.pdf`, `relatorio_2024.pdf`), produzido inteiramente pelo respectivo script Python.

## Estrutura de arquivos

```
estatistica_tavane/
├── Banco_2016.csv           # entrada
├── Banco_2020.csv           # entrada
├── Banco_2024.csv           # entrada
├── analise_2016.py          # script independente
├── analise_2020.py          # script independente
├── analise_2024.py          # script independente
├── relatorio_2016.pdf       # saída
├── relatorio_2020.pdf       # saída
├── relatorio_2024.pdf       # saída
├── skill.md                 # dicionário detalhado das bases
└── CLAUDE.md                # este arquivo
```

**Regra:** os 3 scripts são **totalmente independentes** — sem `utils.py`, sem imports cruzados. Cada um pode ser movido isoladamente para outro lugar e continuar rodando. Duplicação de código entre scripts é aceita e esperada.

## Leitura dos dados

```python
df = pd.read_csv("Banco_AAAA.csv", sep=";", encoding="latin1", decimal=",")
```

Diferenças por ano e detalhes do dicionário estão em [skill.md](skill.md). Pontos críticos:
- 2016 tem ~52% de NA em `RECEITA_TOTAL`.
- 2020 introduz categoria `NÃO DIVULGÁVEL`.
- 2024 tem `TOTAL_VOTOS` como float (1 NA).

## Operacionalizações (definições obrigatórias e idênticas nos 3 anos)

| Conceito | Definição |
|---|---|
| **Branco** | `DS_COR_RACA == "BRANCA"` |
| **Não branco** | `DS_COR_RACA ∈ {"PRETA", "PARDA"}` |
| **Outras raças** | `AMARELA`, `INDÍGENA`, `NÃO DIVULGÁVEL` → **excluídos** das análises raciais (registrar contagem). |
| **Eleito** | `DS_SIT_TOT_TURNO ∈ {"ELEITO POR QP", "ELEITO POR MÉDIA"}` |
| **Candidato competitivo** | `TOTAL_VOTOS ≥ 0,20 × (menor TOTAL_VOTOS entre os eleitos do ano)`. Operacionalmente: ordenar o dataframe por `TOTAL_VOTOS` desc, identificar o último eleito, calcular o limiar, filtrar. |

**Variável binária `RACA_BIN`:** criar coluna com `"Branco"` / `"Não branco"`; descartar linhas fora dessas duas categorias antes de qualquer análise racial.

## Análises a executar (na ordem do relatório)

Cada análise deve produzir **uma tabela numérica + um gráfico + 1–3 linhas de interpretação** no PDF.

1. **Proporção de candidatos brancos vs não-brancos** (universo: todos os candidatos).
2. **Proporção de candidatos competitivos** brancos vs não-brancos.
3. **Proporção de candidatos eleitos** brancos vs não-brancos.
4. **Correlação de Pearson `r`** entre `RECEITA_TOTAL` e `TOTAL_VOTOS`, estratificada por raça (branco / não branco) — universo: todos.
5. **Correlação de Pearson `r`** receita × votos, estratificada por raça — universo: competitivos.
6. **Proporção de competitivos brancos vs não-brancos por decil de receita** (decis calculados sobre `RECEITA_TOTAL` do conjunto de candidatos com receita válida).
7. **Contagem de candidatos brancos vs não-brancos por partido** (universo: todos). Mostrar top 15 partidos por volume; agrupar restantes em "Outros".
8. **Contagem de candidatos competitivos brancos vs não-brancos por partido** (mesma regra de top 15 + "Outros").
9. **Regressão múltipla (OLS) — todos os candidatos**
   - DV: `log1p(TOTAL_VOTOS)`
   - IV: `log1p(RECEITA_TOTAL)`, `DS_GENERO`, `RACA_BIN`, `DS_GRAU_INSTRUCAO`, `ST_REELEICAO`
10. **Regressão logística — todos os candidatos**
    - DV: `ELEITO` (0/1)
    - IV: `log1p(RECEITA_TOTAL)`, `DS_GENERO`, `RACA_BIN`, `ST_REELEICAO`
11. **Regressão múltipla (OLS) — competitivos** — mesmas variáveis do item 9.
12. **Regressão logística — competitivos** — mesmas variáveis do item 10.
13. **Regressão logística com interação `RACA_BIN × log1p(RECEITA_TOTAL)` — todos**.
14. **Regressão logística com interação `RACA_BIN × log1p(RECEITA_TOTAL)` — competitivos**.

### Variáveis "Ver como colocar" — decisões padrão

O usuário marcou bens, ocupação e partido como pendentes nas regressões. Decisões fixas para este trabalho:

- **`BENS_TOTAL`**: aplicar `log1p` e incluir como variável contínua adicional em **versões expandidas** dos modelos (rodar o modelo base + uma versão com `log1p(BENS_TOTAL)`). NAs em bens → imputar 0 ou excluir linha; escolher uma estratégia e documentar no PDF.
- **`DS_OCUPACAO`**: 100+ categorias inviabilizam dummies diretas. **Não incluir** nos modelos principais; opcionalmente apresentar uma seção descritiva separada (top 10 ocupações × eleição). Documentar a exclusão.
- **`SG_PARTIDO`**: incluir como **efeito fixo** apenas em uma versão expandida, agrupando partidos com < 30 candidatos no ano em "Outros".

Sempre rodar **modelo base** (especificado nos itens 9–14) primeiro; modelos expandidos com bens/partido vêm em seções secundárias do relatório.

## Stack obrigatória

| Uso | Biblioteca |
|---|---|
| Dados | `pandas`, `numpy` |
| Estatística inferencial | `statsmodels` (regressões — fornece p-valor, R², AIC, coef) |
| Testes / correlação | `scipy.stats` |
| Gráficos | `matplotlib`, opcionalmente `seaborn` |
| PDF | `matplotlib.backends.backend_pdf.PdfPages` (preferido — simples, sem dependências extras) ou `reportlab` se precisar de tabelas formatadas. **Não usar weasyprint/wkhtmltopdf.** |

## Conteúdo e formato do PDF

Cada `relatorio_AAAA.pdf` deve conter:

1. **Capa** com título, ano da eleição, autor, data de geração.
2. **Seção 0 — Caracterização da amostra**: N total, N excluído por raça fora do recorte, N de NAs em variáveis-chave, definição operacional de competitivo (com valor numérico do limiar).
3. **Seções 1–14** seguindo a ordem das análises, cada uma com:
   - Título descritivo.
   - Tabela de resultados (texto na própria página, gerado via matplotlib `ax.table` ou texto formatado).
   - Gráfico apropriado:
     - Proporções → barras empilhadas ou agrupadas.
     - Correlações → scatter com linha de regressão por raça.
     - Decis de receita → barras lado a lado.
     - Partidos → barras horizontais.
     - Regressões → forest plot de coeficientes com IC95% **ou** tabela formatada de coef/SE/p.
   - Interpretação: 1–3 linhas em texto identificando direção do efeito, significância (p<0.05) e magnitude.
4. **Seção final — Sumário comparativo**: tabela única com os principais coeficientes/correlações para leitura rápida.

### Padrão de gráficos
- Tamanho página A4 retrato (`figsize=(8.27, 11.69)`).
- Cores consistentes: **Branco = `#4C72B0`**, **Não branco = `#DD8452`** em todos os gráficos.
- Fonte mínima 9pt.
- Toda figura com título, eixos rotulados em português, fonte dos dados ("TRE-SP, AAAA") em rodapé.

## Convenções de código

- Cada script começa com bloco docstring identificando ano e objetivo.
- Variáveis em `snake_case`, constantes em `UPPER_CASE`.
- Filtros e transformações documentados com comentário curto **somente quando o "porquê" não for óbvio** (ex.: por que se excluiu uma categoria).
- Todas as transformações `log1p` aplicadas após filtrar NAs nas variáveis monetárias relevantes.
- Modelos `statsmodels`: usar fórmula `patsy` (`smf.ols`, `smf.logit`) para legibilidade.
- Sementes fixadas (`np.random.seed(42)`) caso haja qualquer estocasticidade.
- Cada script imprime no terminal um resumo executivo curto (N, principais coeficientes) ao final, além de gerar o PDF.

## Validações que cada script deve fazer no início

1. Confirmar que o CSV existe; sair com mensagem clara se não.
2. Reportar contagem antes/depois do filtro racial.
3. Reportar contagem de competitivos e o limiar numérico de votos usado.
4. Avisar se alguma regressão não convergiu ou ficou com singularidade.
