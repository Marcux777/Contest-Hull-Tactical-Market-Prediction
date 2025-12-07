# Hull Tactical – Market Prediction

Estamos na **Etapa 1 – Entendimento do problema**.

Este repositório acompanha minha participação na competição **Hull Tactical – Market Prediction** no Kaggle, organizada em etapas de um fluxo típico de projeto em ciência de dados.

---

## 1. Resumo do desafio

Com base na descrição pública do desafio e em materiais externos, o problema é:

- **Domínio:** finanças quantitativas, focado no índice **S&P 500** (mercado de ações dos EUA).
- **História:** a competição tenta desafiar a Hipótese de Mercado Eficiente (EMH), usando ML para prever retornos futuros e gerar uma estratégia tática de alocação em S&P 500.
- **Tipo de tarefa de ML:**
  - No nível de modelagem, é um problema de **regressão de série temporal**: dado o dia *t* e um conjunto de features, o modelo tenta prever o **forward_returns** (retorno excedente do S&P 500 no dia *t+1*).
  - No nível da competição, o modelo precisa devolver, para cada dia, uma **alocação** (exposição entre 0 e 2) em S&P 500 – isto é, quão “comprado” o portfólio fica naquele dia.

### Estrutura dos dados (visão geral)

- Série temporal tabular indexada por `date_id` (cada linha é um dia de negociação).
- Grande número de variáveis agrupadas em famílias:
  - **M\*** – Market / Momentum / Technical
  - **E\*** – Macro Economic
  - **I\*** – Interest Rates
  - **P\*** – Price / Valuation
  - **V\*** – Volatility
  - **S\*** – Sentiment
  - **MOM\*** – indicadores de momentum adicionais
  - **D\*** – variáveis dummy / regime
- Há também uma coluna `is_scored` indicando se aquela linha (dia) é usada na avaliação do leaderboard.

### Alvo, arquivos principais e submissão

- `train.csv`:
  - contém as features (M\*, E\*, I\*, P\*, V\*, S\*, MOM\*, D\*, etc.),
  - colunas de retorno atual / `forward_returns` e outras colunas financeiras (como retorno de mercado e taxa livre de risco),
  - em muitas soluções o alvo é chamado de `target` ou `forward_returns`.
- `test.csv`:
  - contém versões **defasadas** (lagged) das variáveis de retorno/juros,
  - inclui `is_scored`, o que ajuda a evitar vazamento temporal direto do label.
- Formato de submissão:
  - uma coluna de identificação (tipicamente `date_id`),
  - uma coluna de decisão, normalmente chamada `allocation`, que representa a exposição diária sugerida pelo modelo (0 = zerado, 2 = alavancado).

### Métrica da competição

- A leaderboard usa uma **métrica customizada baseada em Sharpe**, muitas vezes chamada *Hull competition Sharpe* ou *Adjusted Sharpe*.
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
   - `train.csv` tem colunas com valor atual de `target` e retorno de mercado.
   - `test.csv` tem versões defasadas dessas variáveis, além de `is_scored`, como mecanismo explícito contra data leakage.
   - Features que usam `target` ou retornos devem ser construídas de forma **causal** (apenas com informação disponível até *t*).

3. **Uso da coluna `is_scored`**
   - No `test`, parte das linhas tem `is_scored = 1` (contam na métrica) e parte `is_scored = 0` (servem de “continuação” da série temporal).
   - Isso impacta a validação local: idealmente o código da métrica precisa replicar essa lógica.

4. **Decisão final é alocação**
   - Fluxo conceitual:
     1. Modelar `forward_returns` (regressão de série temporal);
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

