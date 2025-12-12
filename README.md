# Hull Tactical – Market Prediction

Estamos na **Etapa 1 – Entendimento do problema**.

Este repositório acompanha minha participação na competição **Hull Tactical – Market Prediction** no Kaggle, organizada em etapas de um fluxo típico de projeto em ciência de dados.

---

## 1. Resumo do desafio

Com base na descrição pública do desafio e em materiais externos, o problema é:

- **Domínio:** finanças quantitativas, focado no índice **S&P 500** (mercado de ações dos EUA).
- **História:** a competição tenta desafiar a Hipótese de Mercado Eficiente (EMH), usando ML para prever retornos futuros e gerar uma estratégia tática de alocação em S&P 500.
- **Tipo de tarefa de ML:**
  - No nível de modelagem, é um problema de **regressão de série temporal**: dado o dia *t* e um conjunto de features, o modelo tenta prever o retorno futuro do mercado (no dataset isso aparece como `market_forward_excess_returns` e/ou `forward_returns` com `risk_free_rate`).
  - No nível da competição, o modelo precisa devolver, para cada dia, uma **alocação** (exposição entre 0 e 2) em S&P 500 – isto é, quão “comprado” o portfólio fica naquele dia.

### Metadados oficiais (Kaggle)

- **Página:** https://www.kaggle.com/competitions/hull-tactical-market-prediction
- **Host/Categoria:** Hull Tactical (Featured)
- **Prêmio:** 100,000 USD
- **Deadline:** 2025-12-15 23:59 (UTC)
- **Limites:** 5 submissões/dia; até 5 membros/time
- **Status (consulta 2025-12-12):** 3458 times; topo da leaderboard ~17.507
- **Métrica:** `Hull Competition Sharpe` (custom metric)
- **Tipo de submissão:** kernels-only (Code Competition) — submissão via Notebook/Kernel; a saída é um `submission.parquet` gerado pelo gateway (não há `sample_submission.csv`).

### Estrutura dos dados (visão geral)

- Série temporal tabular indexada por `date_id` (cada linha é um dia de negociação).
- Variáveis agrupadas em famílias (presentes no pacote de dados atual):
  - **D\*** – variáveis dummy / regime (9 colunas)
  - **E\*** – Macro Economic (20 colunas)
  - **I\*** – Interest Rates (9 colunas)
  - **M\*** – Market / Momentum / Technical (18 colunas)
  - **P\*** – Price / Valuation (13 colunas)
  - **S\*** – Sentiment (12 colunas)
  - **V\*** – Volatility (13 colunas)
- No `train.csv` também existem colunas financeiras: `forward_returns`, `risk_free_rate`, `market_forward_excess_returns`.
- No `test.csv`, há `is_scored` (quais linhas contam no score) e colunas `lagged_*` para evitar vazamento temporal do label.

### Alvo, arquivos principais e submissão

- `train.csv`:
  - tamanho (pacote atual): **9021 linhas × 98 colunas** (`date_id` de 0 a 9020),
  - contém as features (D\*, E\*, I\*, M\*, P\*, S\*, V\*) e colunas financeiras (`forward_returns`, `risk_free_rate`, `market_forward_excess_returns`).
- `test.csv`:
  - no pacote atual, é um **sample pequeno** para testes locais: **10 linhas × 99 colunas** (o teste real/oculto é usado no rerun),
  - contém `is_scored` e versões **defasadas** (`lagged_*`) das variáveis de retorno/juros.
- Formato de submissão (Code Competition):
  - o gateway materializa um `submission.parquet` com uma coluna de ID (`date_id` no sample local; pode variar no rerun) e uma coluna `prediction`;
  - a `prediction` representa a decisão do modelo (na prática, a exposição/alocação diária — tipicamente em `[0, 2]`).

### Métrica da competição

- A leaderboard usa a métrica **`Hull Competition Sharpe`** (custom metric baseada em Sharpe).
- Em alto nível, a métrica:
  - constrói uma série de retornos da estratégia usando a **alocação prevista** combinada com o retorno de mercado e a taxa livre de risco;
  - calcula uma espécie de **Sharpe ratio ajustado**, com penalidades específicas para drawdowns / eventos extremos negativos.
- Consequência prática: **minimizar RMSE da previsão de retorno não garante bom score**. É preciso pensar na **função de mapeamento** “previsão de retorno → alocação” levando em conta o comportamento da métrica.

---

## 2. Implicações importantes

Alguns pontos que vão guiar toda a solução:

1. **É série temporal, não tabular iid clássico**
   - Não faz sentido usar `train_test_split` aleatório, pois mistura passado e futuro e gera vazamento temporal.
   - A validação ideal respeita o tempo (splits por blocos de `date_id` ou algo como `TimeSeriesSplit`).

2. **Treino vs. teste com labels defasados**
   - `train.csv` tem colunas de retorno futuro (`forward_returns`, `market_forward_excess_returns`) e taxa livre de risco (`risk_free_rate`) que **não aparecem no `test.csv` da mesma forma**.
   - `test.csv` tem versões defasadas (`lagged_*`) dessas variáveis, além de `is_scored`, como mecanismo explícito contra data leakage.
   - Features que usam `target` ou retornos devem ser construídas de forma **causal** (apenas com informação disponível até *t*).

3. **Uso da coluna `is_scored`**
   - No `test`, parte das linhas tem `is_scored = 1` (contam na métrica) e parte `is_scored = 0` (servem de “continuação” da série temporal).
   - Isso impacta a validação local: idealmente o código da métrica precisa replicar essa lógica.

4. **Decisão final é alocação**
   - Fluxo conceitual:
     1. Modelar `market_forward_excess_returns` (ou derivar de `forward_returns - risk_free_rate`);
     2. Aplicar uma função `g(pred_return)` que converte a previsão em `allocation` em [0, 2];
     3. Avaliar o Sharpe ajustado da estratégia resultante.
   - Competidores fortes tendem a:
     - ajustar bem a função `g` (não apenas “2 se retorno > 0, 0 caso contrário”);
     - usar o código oficial da métrica para **otimizar diretamente** o score out-of-fold.

5. **Boas práticas para este desafio**
   - Usar **validação baseada em tempo** consistente com o período de teste.
   - Implementar o **código oficial da métrica** no notebook para ter métrica local idêntica à da leaderboard.
   - Cuidar de **vazamento temporal** (features com informação futura ou colunas que existem em `train` mas não em `test`).
   - Explorar **feature engineering por famílias** (M\*, E\*, V\*, S\*, etc.) e possivelmente **regimes de mercado** (alta/baixa volatilidade, crise vs expansão).

---

## 3. Próximos passos

Na sequência entraremos na **Etapa 2 – Entendimento dos dados**, analisando:

- tipos de variáveis;
- presença de NaNs;
- distribuição do alvo (`target` / `forward_returns`);
- estrutura temporal (`date_id`);
- primeiras ideias de pré-processamento e esquema de validação específicos para este desafio.

Essa etapa será feita principalmente em notebooks na pasta `notebooks/`, usando os arquivos `train.csv` e `test.csv` do Kaggle.
