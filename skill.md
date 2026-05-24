# Perfil
Estatístico com bastante experiência tanto em temas de estatística quanto python. Estudante de estatística durante sua graduação.

# Bases de dados disponíveis

Três bancos do TRE-SP com candidatos às eleições municipais (vereadores) de São Paulo:

| Arquivo | Ano | Linhas | Colunas |
|---|---|---|---|
| `Banco_2016.csv` | 2016 | 1.205 | 16 |
| `Banco_2020.csv` | 2020 | 1.866 | 16 |
| `Banco_2024.csv` | 2024 | 961  | 17 |

## Como ler os arquivos
- Separador: `;`
- Encoding: `latin1` (ISO-8859-1) — caracteres acentuados ficam quebrados em UTF-8
- Decimal: `,` (vírgula) — usar `decimal=','` no pandas
- Nulos: literal `NA` (pandas reconhece nativamente)

```python
df = pd.read_csv("Banco_2024.csv", sep=";", encoding="latin1", decimal=",")
```

## Dicionário de variáveis (comum aos 3 anos)

| Coluna | Tipo | Descrição |
|---|---|---|
| `SQ_CANDIDATO` | int | Identificador único do candidato (TRE) |
| `NM_CANDIDATO` | str | Nome de registro |
| `NM_SOCIAL_CANDIDATO` | str | Nome social (`#NULO`/`#NULO#` quando ausente) |
| `SG_PARTIDO` | str | Sigla do partido (27–34 partidos distintos por ano) |
| `DS_GENERO` | str | `MASCULINO`, `FEMININO` (em 2020 também `NÃO DIVULGÁVEL`) |
| `DS_GRAU_INSTRUCAO` | str | 7 categorias ordinais (de `LÊ E ESCREVE` a `SUPERIOR COMPLETO`) |
| `DS_ESTADO_CIVIL` | str | `SOLTEIRO(A)`, `CASADO(A)`, `DIVORCIADO(A)`, `SEPARADO(A) JUDICIALMENTE`, `VIÚVO(A)` |
| `DS_COR_RACA` | str | `BRANCA`, `PARDA`, `PRETA`, `AMARELA`, `INDÍGENA` |
| `DS_OCUPACAO` | str | Ocupação declarada (103–130 categorias por ano) |
| `DS_SITUACAO_CANDIDATURA` | str | `APTO` em 2016/2020; `#NE` em 2024 |
| `ST_REELEICAO` | str | `S`/`N` (candidato concorrendo à reeleição) |
| `BENS_TOTAL` | float | Patrimônio declarado (R$) — **com muitos NA** |
| `DS_SIT_TOT_TURNO` | str | Resultado: `ELEITO POR QP`, `ELEITO POR MÉDIA`, `SUPLENTE`, `NÃO ELEITO` |
| `TOTAL_VOTOS` | int/float | Votos nominais recebidos |
| `RECEITA_TOTAL` | float | Receita total da campanha (R$) |
| `TIPO_RECEITA` | str | Lista textual das fontes (próprios, partido, pessoa física, fundo, etc.) |

**Diferença em 2024:** inclui coluna extra `DS_SITUACAO_JULGAMENTO_URNA` (sempre `DEFERIDO` na base atual) e usa `#NE` em vez de `APTO` para `DS_SITUACAO_CANDIDATURA`.

## Dados faltantes (atenção antes de análises)

| Coluna | 2016 | 2020 | 2024 |
|---|---|---|---|
| `BENS_TOTAL` | 467 (38,8%) | 806 (43,2%) | 287 (29,9%) |
| `RECEITA_TOTAL` | 632 (52,4%) | 22 (1,2%) | 10 (1,0%) |
| `TIPO_RECEITA` | 632 (52,4%) | 22 (1,2%) | 10 (1,0%) |
| `TOTAL_VOTOS` | 0 | 0 | 1 |
| `DS_SIT_TOT_TURNO` | 0 | 0 | 1 |

A taxa de NA em `RECEITA_TOTAL` em 2016 é alta — provavelmente reflete candidatos sem prestação de contas registrada no extrato. Considerar imputação por 0, exclusão ou análise estratificada conforme o objetivo.

## Estatísticas descritivas das numéricas

| Variável | Ano | Mín | Máx | Média |
|---|---|---|---|---|
| `TOTAL_VOTOS` | 2016 | 4 | 301.446 | 3.702 |
| `TOTAL_VOTOS` | 2020 | 1 | 167.552 | 2.373 |
| `TOTAL_VOTOS` | 2024 | 1 | 161.386 | 5.318 |
| `RECEITA_TOTAL` | 2016 | 55 | 2,40 M | 65.407 |
| `RECEITA_TOTAL` | 2020 | 0 | 2,57 M | 45.336 |
| `RECEITA_TOTAL` | 2024 | 0 | 4,29 M | 219.723 |
| `BENS_TOTAL` | 2016 | 0,01 | 14,7 M | 468.372 |
| `BENS_TOTAL` | 2020 | 0 | 33,0 M | 533.713 |
| `BENS_TOTAL` | 2024 | 0 | 37,4 M | 808.708 |

Distribuições altamente assimétricas à direita (máximos muito acima da média) — esperar usar log, mediana, ou testes não-paramétricos.

## Diferenças estruturais entre os anos
- **Ordem das colunas** difere entre 2016/2020/2024 (cuidado ao concatenar — usar `pd.concat` com alinhamento por nome).
- **2020** introduz categoria `NÃO DIVULGÁVEL` em gênero, instrução, estado civil, raça e reeleição (LGPD).
- **2024** acrescenta `DS_SITUACAO_JULGAMENTO_URNA` e muda o vocabulário de `DS_SITUACAO_CANDIDATURA`.
- **TIPO_RECEITA** em 2024 usa rótulos diferentes (`FUNDO ESPECIAL`, `OUTROS RECURSOS`) vs. textos longos em 2016/2020.
- Em 2024, `TOTAL_VOTOS` é `float` (por causa do 1 NA); nos demais é `int`.

## Sugestões de análises pertinentes
- **Bivariadas:** receita × votos, bens × votos, escolaridade × eleição, gênero × resultado, raça × votos.
- **Comparação intertemporal:** evolução da participação feminina, distribuição racial, ticket médio de campanha por partido.
- **Modelagem:** regressão logística (eleito vs. não-eleito) com receita, bens, reeleição, escolaridade e gênero como preditores.
- **Testes:** qui-quadrado para tabelas de contingência categóricas; Mann-Whitney/Kruskal-Wallis para numéricas (assimetria forte).
