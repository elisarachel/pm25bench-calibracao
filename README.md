# pm25bench-calibracao

Plugin do [pm25bench](https://github.com/GabrielDSAlves/Benchmark) com dois modelos de
calibração da reanálise do SISAM para o MP2,5 diário da estação CETESB 73 (Congonhas):

| modelo | nome nos resultados | ideia |
|---|---|---|
| `DOM` | `dom` | correção linear local `obs = a + b·rea`, ajustada a cada dia nos dias anteriores do contexto (Monteiro et al., 2013) |
| `GBM` | `gbm` | gradient boosting de árvores (LightGBM) com as mesmas entradas da `linear_regression` do benchmark |
| `GBM(variaveis="estendidas")` | `gbm_estendidas` | o mesmo GBM com defasagens da observação e o viés recente da reanálise |

O plugin não altera nada no benchmark: os modelos entram pelo grupo de entry points
`pm25bench.models`, e a configuração usa os mesmos dados, cortes e métricas do contrato v0.1.

## Instalação

O plugin é instalado no ambiente do benchmark, que fica na pasta vizinha `../Benchmark`.

```bash
cd ../Benchmark
uv sync
# --python garante que o pacote vá para o .venv do benchmark, mesmo com outro venv ativo
uv pip install --python .venv/bin/python -e ../pm25bench-calibracao
.venv/bin/pm25bench models          # deve listar dom e gbm
```

Sem o `--python`, o `uv pip install` instala no venv que estiver ativo no terminal
(variável `VIRTUAL_ENV`), que pode ser de outro projeto.

## Uso

Rode a partir desta pasta, para que os resultados fiquem aqui e não dentro do benchmark:

```bash
cd ../pm25bench-calibracao
../Benchmark/.venv/bin/python -m pytest tests -o addopts=""              # testes
../Benchmark/.venv/bin/pm25bench run configs/benchmark_calibracao_sintetico.yaml
../Benchmark/.venv/bin/pm25bench run configs/benchmark_calibracao.yaml   # dados reais
```

`configs/benchmark_calibracao.yaml` é uma cópia do `configs/benchmark.yaml` oficial em que só
mudam a lista de modelos e a pasta de saída (`results_calibracao/`). Por isso o
`dataset_hash` no `metadata.json` é o mesmo da execução oficial, e as 732 amostras de teste
são as mesmas para todos os modelos.

O notebook `notebooks/diferenca_pareada.ipynb` lê `results_calibracao/predictions.csv` e
calcula o IC 95% da diferença de RMSE entre o `gbm` e os demais modelos, por bootstrap
pareado em blocos de 7 dias, em todas as amostras e só nos dias acima do P90 do treino. Para
abrir o notebook, é preciso ter o `jupyter` no ambiente (grupo `notebook` do
`pyproject.toml`), por exemplo:

```bash
cd notebooks
uv run --no-project --python ../../Benchmark/.venv/bin/python --with jupyter jupyter lab
```

## Os modelos

### DOM

Para cada data de emissão *t*, os pares (`rea_pm2_5`, `obs`) dos dias *t*−6 a *t*−1
formam a janela de ajuste. A reta de mínimos quadrados ordinários tem inclinação
`b = cov(rea, obs) / var(rea)` e intercepto `a = média(obs) − b·média(rea)`, e a previsão é
`a + b·rea_pm2_5(t)`, truncada em zero. A janela anda com *t*, então o modelo se reajusta
todo dia usando só observações até *t*−1, como no sistema operacional.

Se a janela tiver menos de 3 pares válidos ou a reanálise for praticamente constante nela
(o que deixa `var(rea)` perto de zero e a inclinação indefinida), o DOM usa um fator de
escala fixo calculado no treino, `Σ obs / Σ rea_pm2_5`.

**Limitação da janela.** Com `n_context = 7`, o contexto tem só 6 dias de observação antes
de *t*, e a janela do DOM fica com no máximo 6 pares. O sistema operacional usa 14 dias
(`CALIB_NDAYS=14`). Com 6 pontos a inclinação estimada varia muito de um dia para outro, e
o resultado do `dom` neste benchmark representa uma versão mais ruidosa do método em
produção. Avaliar o DOM com 14 dias exigiria `n_context = 15`, uma mudança de contrato que
precisa ser combinada com o autor do benchmark.

O DOM devolve o mesmo valor para todos os leads, porque o contrato não traz previsão da
reanálise para *t*+*k*. Os leads 1 a 3 do `dom` são, portanto, a calibração do dia *t*
usada como previsão dos dias seguintes.

### GBM

Um regressor LightGBM (perda quadrática) por lead. A variante `base` recebe exatamente as
variáveis da `linear_regression` do benchmark: reanálise do dia *t* (6 poluentes),
observação de *t*−1, média da observação no contexto e seno e cosseno do dia do ano. Como
as entradas são as mesmas, a diferença entre `gbm` e `linear_regression` mede o efeito da
não linearidade, e não o efeito de ter mais informação. Um teste automático confere que as
variáveis continuam idênticas às do benchmark.

A variante `estendidas` acrescenta a observação de *t*−2 a *t*−6, o viés aditivo recente
`média(obs − rea_pm2_5)` e a razão recente `Σ obs / Σ rea_pm2_5` no contexto. As duas
últimas dão ao GBM, como preditores, a mesma informação que os métodos de janela (ADD, RAT,
DOM) usam para se reajustar.

O número de árvores é escolhido por parada antecipada na validação (2020–2021): o treino
para quando o erro na validação não melhora por 100 rodadas, e fica a iteração de menor
erro. A validação não entra no ajuste das árvores.

**Por que o GBM não é retreinado em treino + validação.** O número de árvores foi escolhido
para um modelo treinado só com 2010–2019. Retreinar com mais dados e o mesmo número de
árvores não garante o mesmo equilíbrio entre ajuste e sobreajuste, e traria para o ajuste o
período da pandemia, com emissões atípicas. Mantendo a validação fora, o modelo avaliado no
teste é exatamente o que teve o número de árvores escolhido.

A semente é fixa (`seed=42`) e o LightGBM roda em modo determinístico, então duas execuções
dão o mesmo resultado. Para medir a variação entre sementes seria preciso rodar várias
sementes com nomes distintos, o que esta versão não faz: o nome do modelo depende só da
variante.

## Calibração da previsão do CAMS

Com a trilha opcional de previsão do pm25bench (`data.cams_forecast_dir`, branch
`feat/previsao-cams-opcional` do benchmark), o contexto traz as colunas `fc_pm2_5_l{k}`: a
previsão do CAMS emitida às 00 UTC de *t* para o dia *t*+*k*, em média diária na estação.

Os modelos deste experimento usam **só informação disponível na manhã de *t***: a previsão
do CAMS, as observações até *t*−1 e o calendário. A reanálise continua na tabela do
benchmark, mas nenhum deles a lê, porque ela é produzida com meses de atraso e não existe
no momento real da emissão. Um teste automático troca toda a reanálise do contexto por
valores absurdos e confere que nenhuma previsão muda.

| modelo | ideia |
|---|---|
| `dom_previsao` | reta `obs = a + b·fc_l0` ajustada nos dias *t*−6 a *t*−1 (cada dia comparado com a previsão de lead 0 emitida nele), aplicada a `fc_l{k}(t)` em cada lead |
| `gbm_previsao` | LightGBM por lead com a previsão para o dia alvo, o erro recente `média(obs − fc_l0)`, a observação de *t*−1, a média das observações do contexto e a sazonalidade |
| `linear_previsao` | regressão ridge por lead com **as mesmas variáveis** do `gbm_previsao`; a diferença entre os dois isola o efeito da não linearidade |

O `dom_previsao` usa a mesma reta em todos os leads, ou seja, supõe que o viés do CAMS não
depende do lead. O `gbm_previsao` e a `linear_previsao` ajustam um modelo por lead.

### Experimento principal: média de 5 estações

O alvo é a **média de 5 estações** do município de São Paulo, e não uma estação só: 98
Grajaú-Parelheiros, 270 Marginal Tietê, 73 Congonhas, 95 Cidade Universitária e 83
Ibirapuera. São as 5 com pelo menos 78% de dias válidos tanto no treino quanto no teste. O
conjunto é **fixo**: se a média usasse as estações disponíveis em cada dia, a composição
mudaria entre treino e teste (6 estações do município só começam em 2023), e o modelo
confundiria essa mudança com mudança de viés do CAMS.

- Observação: média horária das estações válidas, só nas horas com pelo menos 4 das 5;
  depois a média diária com a regra das 18 horas do benchmark.
- Previsão do CAMS: interpolada em cada uma das 5 estações e tirada a média hora a hora,
  antes da média diária (`benchmark_cams_forecast/data/media_sp5/`).
- A média se aproxima mais da escala da célula do CAMS (0,4°) do que um ponto, e reduz o
  erro de representatividade de uma estação isolada.

O `pm25bench run` trabalha com uma estação só, então este experimento roda pelo script
`scripts/rodar_media.py`, que monta a tabela (`pm25bench_calibracao.media`) e chama
`run_benchmark` com o mesmo protocolo:

```bash
../Benchmark/.venv/bin/python scripts/rodar_media.py configs/benchmark_media_sp5_previsao.yaml
../Benchmark/.venv/bin/python scripts/figuras.py configs/benchmark_media_sp5_previsao.yaml
```

O primeiro grava `results_media_sp5_previsao/`; o segundo, as figuras do relatório em
`figuras/` (cada uma responde a uma pergunta, listadas no início do script) e as tabelas
`diferenca_pareada.csv` (bootstrap pareado), `skill.csv` (MSESS contra a climatologia e o
CAMS bruto, com IC) e `metricas_categoricas.csv` (POD, FAR, CSI, ETS, viés de frequência e
acurácia nos limiares 15, 25 e 50 µg/m³) em `results_media_sp5_previsao/`. O notebook
`notebooks/diferenca_pareada.ipynb` explica e calcula o bootstrap pareado passo a passo.

Sensibilidade à semente e custo computacional (grava `sementes.csv` e `tempos.csv`):

```bash
../Benchmark/.venv/bin/python scripts/sementes_tempos.py configs/benchmark_media_sp5_previsao.yaml
```

O `gbm_previsao` padrão não sorteia nada no treino e não depende da semente. O script
confirma isso e treina uma variante com subamostragem (80% das amostras e das variáveis por
árvore, opções `bagging_fraction` e `feature_fraction` do `GBM`) com 10 sementes.

Dados necessários: previsões em `../benchmark_cams_forecast/data`, reanálise em
`../dados_benchmark/sisam` (com 2025) e observações em `../benchmark_pm2_5/data`.

A config sintética (sem baixar dados) continua valendo para testar os modelos:

```bash
../Benchmark/.venv/bin/pm25bench run configs/benchmark_calibracao_previsao_sintetico.yaml
```

## Estrutura

```
src/pm25bench_calibracao/
├── features.py   # features_base, features_estendidas, features_previsao e nomes
├── linear.py     # LinearPrevisao
├── media.py      # tabela da média de estações (ESTACOES_SP5)
├── avaliacao.py  # bootstrap pareado e tabela de diferenças
├── dom.py        # DOM e DOMPrevisao
└── gbm.py        # GBM e GBMPrevisao (com importancias() por ganho)
scripts/rodar_media.py, figuras.py
figuras/                       # figuras do relatório
tests/test_calibracao.py
configs/benchmark_calibracao.yaml, benchmark_calibracao_sintetico.yaml
configs/benchmark_media_sp5_previsao.yaml, benchmark_calibracao_previsao_sintetico.yaml
notebooks/diferenca_pareada.ipynb
```
