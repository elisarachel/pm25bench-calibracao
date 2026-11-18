"""Avaliação dos modelos: bootstrap pareado em blocos, skill score e métricas categóricas."""

from __future__ import annotations

# Callable: tipos das funções de métrica e de estatística
from collections.abc import Callable, Sequence

# numpy: sorteios e quantis
import numpy as np

# pandas: tabelas de previsões
import pandas as pd

# métricas categóricas do benchmark (mesma implementação do runner)
from pm25bench.metrics import contingency, csi, ets, far, frequency_bias, pod

#: Métrica: recebe (obs, previsão) e devolve um número.
Metrica = Callable[[np.ndarray, np.ndarray], float]

#: Estatística de dois modelos: recebe (obs, previsão de A, previsão de B).
Estatistica = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def bootstrap_pareado(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    estatistica: Estatistica,
    block_days: int = 7,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """IC 95% de uma estatística de dois modelos por bootstrap pareado em blocos.

    O bootstrap aproxima a distribuição da estatística reamostrando os dados com reposição
    e recalculando-a em cada reamostra; os quantis de 2,5% e 97,5% formam o intervalo
    (método percentil). Como o MP2,5 diário é autocorrelacionado, a reamostragem é por
    blocos de ``block_days`` dias consecutivos (rótulo ``dias desde 1970 // block_days``, a
    mesma regra do runner do benchmark). Em cada reamostra, **os mesmos blocos** entram nos
    dois modelos: como os dois erram mais nos mesmos dias, a covariância positiva entre os
    erros estreita o intervalo, por exemplo ``Var(A - B) = Var(A) + Var(B) - 2·Cov(A, B)``.

    Parameters
    ----------
    df_a, df_b : pandas.DataFrame
        Previsões de dois modelos (colunas ``issue_date``, ``lead``, ``obs``, ``pred``).
    estatistica : callable
        Função ``estatistica(obs, pred_a, pred_b) -> float``.
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
    pontual = estatistica(o, fa, fb)
    # gerador com semente fixa (reprodutível)
    rng = np.random.default_rng(seed)
    # réplicas da estatística
    reps = np.empty(n_boot)
    # cada iteração é uma reamostra
    for i in range(n_boot):
        # sorteia tantos blocos quantos existem, com reposição
        sorteio = rng.integers(0, len(grupos), len(grupos))
        # junta as amostras dos blocos sorteados
        idx = np.concatenate([grupos[j] for j in sorteio])
        # estatística nas MESMAS amostras para os dois modelos
        reps[i] = estatistica(o[idx], fa[idx], fb[idx])
    # quantis de 2,5% e 97,5% (intervalo percentil de 95%), ignorando réplicas indefinidas
    lo, hi = np.nanquantile(reps, [0.025, 0.975])
    # devolve estimativa e limites
    return float(pontual), float(lo), float(hi)


def bootstrap_diferenca_pareada(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metrica: Metrica,
    block_days: int = 7,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """IC 95% da diferença ``metrica(A) - metrica(B)`` (ver ``bootstrap_pareado``)."""

    # diferença da métrica entre os dois modelos, nas mesmas amostras
    def diferenca(o: np.ndarray, fa: np.ndarray, fb: np.ndarray) -> float:
        # métrica de A menos métrica de B
        return metrica(o, fa) - metrica(o, fb)

    # bootstrap pareado da diferença
    return bootstrap_pareado(df_a, df_b, diferenca, block_days, n_boot, seed)


def msess(o: np.ndarray, f: np.ndarray, ref: np.ndarray) -> float:
    """Skill score do erro quadrático médio: ``1 - MSE(f) / MSE(ref)``.

    Vale 1 para a previsão perfeita, 0 quando o método erra tanto quanto a referência e é
    negativo quando erra mais. Como é uma razão de erros nas mesmas amostras, mede o ganho
    relativo sobre a referência (por exemplo, 0,4 = 40% menos erro quadrático).
    """
    # erro quadrático médio do método
    mse_f = float(np.mean((f - o) ** 2))
    # erro quadrático médio da referência
    mse_ref = float(np.mean((ref - o) ** 2))
    # razão indefinida se a referência for perfeita
    return 1.0 - mse_f / mse_ref if mse_ref > 0 else float("nan")


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


def tabela_skill(
    previsoes: pd.DataFrame,
    modelos: Sequence[str],
    referencias: Sequence[str],
    block_days: int = 7,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """MSESS de cada modelo contra cada referência, por lead, com IC 95% pareado.

    Returns
    -------
    pandas.DataFrame
        Colunas ``modelo``, ``referencia``, ``lead``, ``msess``, ``ic_lo``, ``ic_hi``.
    """
    # linhas da tabela final
    linhas = []
    # percorre os leads
    for lead in sorted(previsoes["lead"].unique()):
        # previsões deste lead
        sub = previsoes[previsoes["lead"] == lead]
        # percorre as referências
        for ref_nome in referencias:
            # previsões da referência
            ref = sub[sub["model"] == ref_nome]
            # percorre os modelos (a referência contra si mesma não interessa)
            for nome in modelos:
                # pula a própria referência
                if nome == ref_nome:
                    continue
                # previsões do modelo
                mod = sub[sub["model"] == nome]
                # skill score com IC pareado
                s, lo, hi = bootstrap_pareado(mod, ref, msess, block_days, n_boot, seed)
                # guarda a linha
                linhas.append(
                    {
                        # modelo avaliado
                        "modelo": nome,
                        # referência do skill score
                        "referencia": ref_nome,
                        # lead em dias
                        "lead": int(lead),
                        # skill score pontual
                        "msess": s,
                        # limite inferior do IC 95%
                        "ic_lo": lo,
                        # limite superior do IC 95%
                        "ic_hi": hi,
                    }
                )
    # devolve como DataFrame
    return pd.DataFrame(linhas)


def tabela_categorica(previsoes: pd.DataFrame, limiares: Sequence[float]) -> pd.DataFrame:
    """Métricas de ultrapassagem de limiar por modelo, lead e limiar.

    Um dia é "evento" quando o MP2,5 diário é ``>= limiar``. Da tabela de contingência
    (a = acertos, b = falsos alarmes, c = perdas, d = negativos corretos):

    - POD = a / (a + c): fração dos eventos observados que foi prevista;
    - FAR = b / (a + b): fração dos eventos previstos que não aconteceu;
    - CSI = a / (a + b + c): acertos sobre todos os casos em que evento foi previsto ou
      observado (ignora os negativos corretos);
    - ETS: CSI descontando os acertos esperados ao acaso;
    - viés de frequência = (a + b) / (a + c): > 1 prevê eventos demais, < 1 de menos;
    - acurácia = (a + d) / n: proporção de dias classificados certo. Com eventos raros ela
      é dominada pelos negativos corretos e fica alta mesmo sem nenhum acerto de evento.

    Returns
    -------
    pandas.DataFrame
        Uma linha por modelo, lead e limiar, com o número de eventos observados.
    """
    # linhas da tabela
    linhas = []
    # percorre modelo e lead
    for (nome, lead), g in previsoes.groupby(["model", "lead"]):
        # observação e previsão
        o, f = g["obs"].to_numpy(), g["pred"].to_numpy()
        # percorre os limiares
        for thr in limiares:
            # tabela de contingência
            a, b, c, d = contingency(o, f, thr)
            # guarda a linha
            linhas.append(
                {
                    # identificação
                    "modelo": nome,
                    "lead": int(str(lead)),
                    "limiar": thr,
                    # número de dias com evento observado
                    "eventos_obs": a + c,
                    # número de dias com evento previsto
                    "eventos_prev": a + b,
                    # métricas
                    "pod": pod(o, f, thr),
                    "far": far(o, f, thr),
                    "csi": csi(o, f, thr),
                    "ets": ets(o, f, thr),
                    "vies_freq": frequency_bias(o, f, thr),
                    "acuracia": (a + d) / (a + b + c + d),
                }
            )
    # devolve como DataFrame
    return pd.DataFrame(linhas)
