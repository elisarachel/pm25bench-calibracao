"""Comparação de modelos: IC da diferença de uma métrica por bootstrap pareado em blocos."""

from __future__ import annotations

# Callable: tipo da função de métrica
from collections.abc import Callable

# numpy: sorteios e quantis
import numpy as np

# pandas: tabelas de previsões
import pandas as pd

#: Métrica: recebe (obs, previsão) e devolve um número.
Metrica = Callable[[np.ndarray, np.ndarray], float]


def bootstrap_diferenca_pareada(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metrica: Metrica,
    block_days: int = 7,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """IC 95% da diferença ``metrica(A) - metrica(B)`` por bootstrap pareado em blocos.

    O bootstrap aproxima a distribuição da diferença reamostrando os dados com reposição e
    recalculando a diferença em cada reamostra; os quantis de 2,5% e 97,5% formam o
    intervalo (método percentil). Como o MP2,5 diário é autocorrelacionado, a reamostragem
    é por blocos de ``block_days`` dias consecutivos (rótulo ``dias desde 1970 //
    block_days``, a mesma regra do runner do benchmark). Em cada reamostra, **os mesmos
    blocos** entram nos dois modelos: como os dois erram mais nos mesmos dias, a covariância
    positiva entre os erros estreita o intervalo da diferença,
    ``Var(A - B) = Var(A) + Var(B) - 2·Cov(A, B)``.

    Parameters
    ----------
    df_a, df_b : pandas.DataFrame
        Previsões de dois modelos (colunas ``issue_date``, ``lead``, ``obs``, ``pred``).
    metrica : callable
        Função ``metrica(obs, pred) -> float``.
    block_days : int
        Tamanho dos blocos de dias consecutivos.
    n_boot : int
        Número de reamostras.
    seed : int
        Semente do gerador aleatório.

    Returns
    -------
    tuple of float
        (estimativa pontual, limite de 2,5%, limite de 97,5%).

    Raises
    ------
    ValueError
        Se os dois modelos não tiverem exatamente as mesmas amostras.
    """
    # alinha os dois modelos pelas mesmas amostras (data de emissão e lead)
    par = df_a.merge(df_b, on=["issue_date", "lead"], suffixes=("_a", "_b"))
    # as duas tabelas precisam ter as mesmas amostras
    if len(par) != len(df_a) or len(par) != len(df_b):
        raise ValueError("os dois modelos não têm as mesmas amostras")
    # observação comum
    o = par["obs_a"].to_numpy(dtype="float64")
    # previsão de A
    fa = par["pred_a"].to_numpy(dtype="float64")
    # previsão de B
    fb = par["pred_b"].to_numpy(dtype="float64")
    # dias desde 1970 de cada data de emissão
    dias = (pd.to_datetime(par["issue_date"]) - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    # rótulo do bloco, igual ao do runner
    _, inverso = np.unique(dias // block_days, return_inverse=True)
    # posições das amostras de cada bloco
    grupos = [np.flatnonzero(inverso == i) for i in range(inverso.max() + 1)]
    # estimativa pontual com todas as amostras
    pontual = metrica(o, fa) - metrica(o, fb)
    # gerador com semente fixa (reprodutível)
    rng = np.random.default_rng(seed)
    # réplicas da diferença
    reps = np.empty(n_boot)
    # cada iteração é uma reamostra
    for i in range(n_boot):
        # sorteia tantos blocos quantos existem, com reposição
        sorteio = rng.integers(0, len(grupos), len(grupos))
        # junta as amostras dos blocos sorteados
        idx = np.concatenate([grupos[j] for j in sorteio])
        # diferença calculada nas MESMAS amostras para os dois modelos
        reps[i] = metrica(o[idx], fa[idx]) - metrica(o[idx], fb[idx])
    # quantis de 2,5% e 97,5% (intervalo percentil de 95%)
    lo, hi = np.quantile(reps, [0.025, 0.975])
    # devolve estimativa e limites
    return float(pontual), float(lo), float(hi)


def tabela_diferencas(
    previsoes: pd.DataFrame,
    referencia: str,
    comparados: list[str],
    metrica: Metrica,
    block_days: int = 7,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Diferença ``metrica(A) - metrica(referência)`` por lead, com IC 95% pareado.

    Uma diferença positiva quer dizer que a referência erra menos que A (para métricas de
    erro como RMSE). A diferença é significativa a 95% quando o intervalo não contém zero.
    """
    # linhas da tabela final
    linhas = []
    # percorre os leads em ordem
    for lead in sorted(previsoes["lead"].unique()):
        # previsões deste lead
        sub = previsoes[previsoes["lead"] == lead]
        # previsões da referência neste lead
        ref = sub[sub["model"] == referencia]
        # percorre os modelos comparados
        for nome in comparados:
            # previsões do modelo A neste lead
            mod = sub[sub["model"] == nome]
            # estimativa e IC com os mesmos parâmetros de bootstrap do benchmark
            d, lo, hi = bootstrap_diferenca_pareada(mod, ref, metrica, block_days, n_boot, seed)
            # guarda a linha
            linhas.append(
                {
                    # modelo comparado
                    "modelo": nome,
                    # lead em dias
                    "lead": int(lead),
                    # número de amostras
                    "n": len(ref),
                    # diferença pontual
                    "dif": d,
                    # limite inferior do IC 95%
                    "ic_lo": lo,
                    # limite superior do IC 95%
                    "ic_hi": hi,
                    # significativa se o intervalo não contém zero
                    "significativa": not (lo <= 0 <= hi),
                }
            )
    # devolve como DataFrame
    return pd.DataFrame(linhas)
