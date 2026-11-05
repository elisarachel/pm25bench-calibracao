"""Variáveis de entrada (preditores) extraídas do contexto de uma amostra do pm25bench.

As duas funções recebem o ``context`` de uma amostra (``n_context`` linhas, dias t-n+1 ... t)
e devolvem um vetor 1D de ``float64``. Nenhuma delas lê a ``obs`` da última linha (dia t),
que no benchmark é ``NaN`` porque o dia ainda não terminou: ler essa célula seria vazamento.
"""

from __future__ import annotations

# numpy: álgebra com vetores
import numpy as np

# pandas: o contexto chega como DataFrame
import pandas as pd

# TARGET é o nome da coluna observada ("obs"); REANALYSIS_FEATURES são as 6 colunas rea_*
from pm25bench.data import REANALYSIS_FEATURES, TARGET

#: Defasagens individuais da observação usadas nas variáveis estendidas (t-2 ... t-6).
DEFASAGENS: list[int] = [2, 3, 4, 5, 6]

#: Nomes das variáveis de ``features_base``, na mesma ordem do vetor.
NOMES_BASE: list[str] = [
    # as 6 colunas de reanálise do dia t, na ordem fixa do benchmark
    *REANALYSIS_FEATURES,
    # observação do dia t-1 (persistência)
    "obs_t-1",
    # média da observação de t-n+1 a t-1
    "obs_media_contexto",
    # sazonalidade: seno do dia do ano
    "sazonal_sin",
    # sazonalidade: cosseno do dia do ano
    "sazonal_cos",
]

#: Nomes das variáveis de ``features_estendidas``, na mesma ordem do vetor.
NOMES_ESTENDIDAS: list[str] = [
    # começa com as variáveis base
    *NOMES_BASE,
    # uma coluna por defasagem: obs_t-2, ..., obs_t-6
    *[f"obs_t-{k}" for k in DEFASAGENS],
    # viés aditivo recente: média de (obs - rea_pm2_5) no contexto
    "vies_recente",
    # razão recente: soma(obs) / soma(rea_pm2_5) no contexto
    "razao_recente",
]


def features_base(context: pd.DataFrame) -> np.ndarray:
    """Mesmas variáveis da ``linear_regression`` do benchmark, sem o intercepto.

    Reproduz ``pm25bench.models.linear._features`` sem o 1.0 inicial (o intercepto não
    serve para árvores de decisão, que já têm um termo constante implícito). Assim o GBM e
    a regressão linear recebem as mesmas entradas, e a diferença de desempenho entre eles
    isola o efeito da não linearidade.

    Parameters
    ----------
    context : pandas.DataFrame
        Contexto de uma amostra, com ``obs`` e ``rea_*`` e índice diário (t-n+1 ... t).

    Returns
    -------
    numpy.ndarray
        Vetor ``[rea_*(t) (6), obs(t-1), média(obs[t-n+1 .. t-1]), sin(2π·doy/365,25),
        cos(2π·doy/365,25)]``, com ``doy`` = dia do ano de t.
    """
    # reanálise da última linha (dia t), nas 6 colunas e na ordem fixa do benchmark
    rea_t = context[REANALYSIS_FEATURES].iloc[-1].to_numpy(dtype="float64")
    # observações de t-n+1 a t-1: todas as linhas menos a última (dia t nunca é lido)
    passado = context[TARGET].iloc[:-1].to_numpy(dtype="float64")
    # dia do ano (1 a 366) da data de emissão t
    doy = context.index[-1].day_of_year
    # posição no ciclo anual como ângulo; seno e cosseno juntos evitam o salto 31/12 -> 01/01
    sazonal = [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    # concatena na mesma ordem de _features (sem o intercepto)
    return np.concatenate([rea_t, [passado[-1], passado.mean()], sazonal])


def features_estendidas(context: pd.DataFrame) -> np.ndarray:
    """Variáveis base mais defasagens da observação e o viés recente da reanálise.

    Acrescenta a ``features_base``:

    - ``obs`` de t-2 a t-6, cada uma numa coluna (defasagens);
    - viés aditivo recente, ``média(obs - rea_pm2_5)`` de t-n+1 a t-1, o mesmo que o
      método ADD usaria como correção;
    - razão recente, ``soma(obs) / soma(rea_pm2_5)`` de t-n+1 a t-1, o mesmo que o método
      RAT usaria como fator multiplicativo.

    As duas últimas dão ao modelo, na forma de preditor, a informação que os métodos de
    janela móvel (ADD, RAT, DOM) usam para se reajustar.

    Parameters
    ----------
    context : pandas.DataFrame
        Contexto com pelo menos 7 linhas (6 dias de observação antes de t).

    Returns
    -------
    numpy.ndarray
        Vetor na ordem de ``NOMES_ESTENDIDAS``. A razão recente vale ``NaN`` se a soma da
        reanálise for zero (o LightGBM trata ``NaN`` como valor ausente).

    Raises
    ------
    ValueError
        Se o contexto tiver menos linhas do que as defasagens exigem.
    """
    # a defasagem t-6 exige 6 dias antes de t, mais o próprio t: 7 linhas
    if len(context) < max(DEFASAGENS) + 1:
        # avisa cedo em vez de devolver um vetor com tamanho errado
        raise ValueError(f"features_estendidas exige n_context >= {max(DEFASAGENS) + 1}")
    # parte comum com a regressão do benchmark
    base = features_base(context)
    # observações de t-n+1 a t-1 (a linha de t fica de fora)
    obs = context[TARGET].iloc[:-1].to_numpy(dtype="float64")
    # reanálise de MP2,5 nos mesmos dias t-n+1 a t-1, pareada com obs
    rea = context["rea_pm2_5"].iloc[:-1].to_numpy(dtype="float64")
    # obs[-k] é o dia t-k, porque obs[-1] é t-1
    defasagens = [obs[-k] for k in DEFASAGENS]
    # viés aditivo médio da reanálise no contexto (positivo = reanálise subestima)
    vies = float(np.mean(obs - rea))
    # denominador da razão: soma da reanálise no contexto
    soma_rea = float(np.sum(rea))
    # razão obs/reanálise; NaN quando a soma é zero para não dividir por zero
    razao = float(np.sum(obs)) / soma_rea if soma_rea != 0.0 else float("nan")
    # junta tudo na ordem de NOMES_ESTENDIDAS
    return np.concatenate([base, defasagens, [vies, razao]])
