"""Sensibilidade do LightGBM à semente e custo computacional dos métodos.

Uso (da raiz do plugin, no ambiente do benchmark)::

    ../Benchmark/.venv/bin/python scripts/sementes_tempos.py \
        configs/benchmark_media_sp5_previsao.yaml

1. **Sementes.** O ``gbm_previsao`` padrão não sorteia nada no treino e não depende da
   semente (verificado aqui). Para medir a sensibilidade à aleatoriedade, treina-se uma
   variante com subamostragem (80% das amostras e 80% das variáveis por árvore) com 10
   sementes e compara-se a variação entre sementes com a diferença para a regressão linear.
2. **Tempos.** Para cada método: tempo de treino (mediana de 3 repetições) e tempo de
   previsão por amostra (uma amostra = uma rodada diária, com os 4 leads).

Grava ``sementes.csv`` e ``tempos.csv`` em ``<output_dir>/``.
"""

from __future__ import annotations

# argparse: leitura do caminho da config
import argparse

# os: número de núcleos da máquina
import os

# platform: descrição da máquina
import platform

# time: cronômetro de alta resolução
import time

# Path: caminhos
from pathlib import Path

# numpy: contas
import numpy as np

# pandas: tabelas
import pandas as pd

# API do benchmark
from pm25bench import load_config, load_model, make_samples, split_by_time

# métricas iguais às do runner
from pm25bench.metrics import mae, rmse

# modelo e tabela da média das estações
from pm25bench_calibracao import GBMPrevisao
from pm25bench_calibracao.media import tabela_media

#: Leads avaliados.
LEADS = [0, 1, 2, 3]

#: Sementes da variante com subamostragem.
SEMENTES = list(range(10))

#: Métodos cronometrados (nomes registrados no pm25bench).
METODOS = [
    "persistence",
    "climatology",
    "forecast_raw",
    "dom_previsao",
    "linear_previsao",
    "gbm_previsao",
]


def prever(modelo: object, amostras: list) -> np.ndarray:  # type: ignore[type-arg]
    """Previsões de todas as amostras: matriz (amostras × leads)."""
    # uma linha por amostra
    return np.array([modelo.predict(s.context, LEADS) for s in amostras])  # type: ignore[attr-defined]


def metricas(pred: np.ndarray, alvo: np.ndarray) -> dict[str, float]:
    """RMSE e MAE por lead, num dicionário ``{rmse_l0: ..., mae_l0: ...}``."""
    # resultado
    out = {}
    # percorre os leads
    for j, lead in enumerate(LEADS):
        # RMSE do lead
        out[f"rmse_l{lead}"] = rmse(alvo[:, j], pred[:, j])
        # MAE do lead
        out[f"mae_l{lead}"] = mae(alvo[:, j], pred[:, j])
    # devolve
    return out


def main() -> None:
    """Roda as duas análises e grava as tabelas."""
    # analisador de argumentos
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # config do experimento
    p.add_argument("config", type=Path)
    # lê a linha de comando
    a = p.parse_args()
    # config
    cfg = load_config(a.config)
    # tabela da média das estações
    df = tabela_media(
        cfg.data.sisam_dir,
        cfg.data.cetesb_dir,
        cfg.data.cams_forecast_dir,  # type: ignore[arg-type]
        min_hours=cfg.data.min_hours,
    )
    # divisão do benchmark
    treino, val, teste = split_by_time(df, cfg.data.train_end, cfg.data.val_end, cfg.data.gap_days)
    # amostras de teste (as mesmas do runner)
    amostras = list(make_samples(teste, cfg.n_context, LEADS))
    # alvos (amostras × leads)
    alvo = np.array([s.target for s in amostras])
    # pasta de resultados
    saida = Path(cfg.output_dir)

    # ------------------------------------------------------------ 1. sementes
    # referência linear (determinística)
    lin = load_model("linear_previsao")
    # ajuste
    lin.fit(treino, val)
    # métricas da linear
    m_lin = metricas(prever(lin, amostras), alvo)
    # GBM padrão com duas sementes: confirma que não depende da semente
    g0, g1 = GBMPrevisao(seed=0), GBMPrevisao(seed=1)
    # ajustes
    g0.fit(treino, val)
    # segundo ajuste
    g1.fit(treino, val)
    # maior diferença entre as previsões das duas sementes
    dif_padrao = float(np.abs(prever(g0, amostras) - prever(g1, amostras)).max())
    # registro
    print(f"GBM padrão: diferença máxima entre sementes = {dif_padrao:.2e} µg/m³")
    # linhas da tabela de sementes
    linhas = []
    # percorre as sementes da variante com subamostragem
    for seed in SEMENTES:
        # variante estocástica
        g = GBMPrevisao(seed=seed, bagging_fraction=0.8, feature_fraction=0.8)
        # ajuste
        g.fit(treino, val)
        # métricas desta semente
        m = metricas(prever(g, amostras), alvo)
        # diferença RMSE(linear) - RMSE(GBM) por lead (> 0: GBM erra menos)
        m.update({f"dif_lin_l{k}": m_lin[f"rmse_l{k}"] - m[f"rmse_l{k}"] for k in LEADS})
        # guarda com a semente
        linhas.append({"semente": seed, **m})
    # tabela por semente
    sem = pd.DataFrame(linhas)
    # grava
    sem.round(4).to_csv(saida / "sementes.csv", index=False)
    # resumo: média, desvio, mínimo e máximo entre sementes
    resumo = sem.drop(columns="semente").agg(["mean", "std", "min", "max"]).T.round(3)
    # mostra
    print("\nvariante com subamostragem, 10 sementes:")
    # tabela do resumo
    print(resumo.to_string())
    # RMSE da linear para comparação
    print("\nregressão linear (determinística):", {k: round(v, 3) for k, v in m_lin.items()})

    # ------------------------------------------------------------ 2. tempos
    # linhas da tabela de tempos
    tempos = []
    # percorre os métodos
    for nome in METODOS:
        # tempos de treino de 3 repetições
        t_fit = []
        # repetições
        for _ in range(3):
            # modelo novo
            mod = load_model(nome)
            # início
            t0 = time.perf_counter()
            # treino
            mod.fit(treino, val)
            # duração
            t_fit.append(time.perf_counter() - t0)
        # início da previsão
        t0 = time.perf_counter()
        # previsão de todas as amostras de teste
        prever(mod, amostras)
        # duração por amostra
        t_pred = (time.perf_counter() - t0) / len(amostras)
        # guarda
        tempos.append(
            {
                # método
                "modelo": nome,
                # mediana do treino, em segundos
                "treino_s": float(np.median(t_fit)),
                # previsão de uma rodada (4 leads), em milissegundos
                "previsao_ms_por_rodada": t_pred * 1000,
            }
        )
    # tabela de tempos
    tab = pd.DataFrame(tempos)
    # grava
    tab.round(5).to_csv(saida / "tempos.csv", index=False)
    # máquina
    print(f"\nmáquina: {platform.processor() or platform.machine()}, {os.cpu_count()} núcleos")
    # tempos
    print(tab.round(4).to_string(index=False))


# executa só quando chamado como script
if __name__ == "__main__":
    main()
