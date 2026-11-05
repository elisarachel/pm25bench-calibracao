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

## Estrutura

```
src/pm25bench_calibracao/
├── features.py   # features_base, features_estendidas, NOMES_BASE, NOMES_ESTENDIDAS
├── dom.py        # DOM
└── gbm.py        # GBM (com importancias() por ganho)
tests/test_calibracao.py
configs/benchmark_calibracao.yaml, benchmark_calibracao_sintetico.yaml
notebooks/diferenca_pareada.ipynb
```
