"""Tabela do benchmark para a média de várias estações CETESB (uma "estação virtual").

O pm25bench trabalha com uma estação. Este módulo monta a mesma tabela diária que
``pm25bench.build_dataset``, trocando a observação de uma estação pela média de um conjunto
fixo de estações, e a previsão do CAMS na estação pela previsão média nas mesmas estações.
A tabela é passada a ``run_benchmark(cfg, df)``, então o protocolo (divisão, amostras,
métricas) continua o do benchmark.
"""

from __future__ import annotations

# Sequence: tipo das listas de estações
from collections.abc import Sequence

# Path: caminhos das pastas de dados
from pathlib import Path

# pandas: séries e tabelas
import pandas as pd

# leitura dos dados no formato do benchmark
from pm25bench.data import (
    TARGET,
    daily_mean,
    read_cams_forecast,
    read_sisam,
    read_station_hourly,
    sisam_files,
    station_cd_mun,
)

#: Estações do município de São Paulo usadas na média: as 5 com pelo menos 78% de dias
#: válidos tanto no treino (2015-2021) quanto no teste (2023-2025). O conjunto é fixo para
#: que a média não mude de composição entre treino e teste.
ESTACOES_SP5: list[int] = [98, 270, 73, 95, 83]

#: Mínimo de estações com dado válido numa hora para a média horária valer.
MIN_ESTACOES: int = 4

#: Nome da pasta da previsão média em ``benchmark_cams_forecast/data``.
NOME_SP5 = "media_sp5"


def obs_media_horaria(
    cetesb_dir: str | Path, estacoes: Sequence[int], min_estacoes: int = MIN_ESTACOES
) -> pd.Series:
    """Média horária de MP2,5 entre estações, só nas horas com estações suficientes.

    Em cada hora, a média usa as estações com dado válido; se forem menos de
    ``min_estacoes``, a hora fica sem valor. A média diária vem depois, com a regra das 18
    horas do benchmark (``daily_mean``), aplicada à série média.

    Parameters
    ----------
    cetesb_dir : path
        Pasta com uma subpasta por estação (``<código>/<ano>.csv``).
    estacoes : sequence of int
        Códigos das estações.
    min_estacoes : int
        Mínimo de estações válidas por hora.

    Returns
    -------
    pandas.Series
        MP2,5 horário médio em µg/m³, no horário original da CETESB.
    """
    # uma coluna por estação, alinhadas pelo horário
    horario = pd.concat({e: read_station_hourly(cetesb_dir, e) for e in estacoes}, axis=1)
    # quantas estações têm dado em cada hora
    n = horario.notna().sum(axis=1)
    # média entre as estações válidas, só nas horas com estações suficientes
    media = horario.mean(axis=1).where(n >= min_estacoes)
    # nome igual ao das séries de uma estação
    media.name = TARGET
    # devolve só as horas com valor
    return media.dropna()


def tabela_media(
    sisam_dir: str | Path,
    cetesb_dir: str | Path,
    cams_forecast_dir: str | Path | None = None,
    estacoes: Sequence[int] = ESTACOES_SP5,
    nome_previsao: str = NOME_SP5,
    min_estacoes: int = MIN_ESTACOES,
    min_hours: int = 18,
) -> pd.DataFrame:
    """Tabela diária do benchmark com a média das estações.

    Mesmas colunas e regras de ``pm25bench.build_dataset``: ``obs`` (média das estações),
    ``rea_*`` (reanálise do município) e, se ``cams_forecast_dir`` for informado, as
    colunas ``fc_*`` lidas de ``<cams_forecast_dir>/<nome_previsao>/``.

    Raises
    ------
    ValueError
        Se as estações não estiverem todas no mesmo município (a reanálise é municipal).
    """
    # município de cada estação
    municipios = {station_cd_mun(e) for e in estacoes}
    # a reanálise é por município: as estações precisam estar no mesmo
    if len(municipios) != 1:
        raise ValueError(f"estações em municípios diferentes: {sorted(municipios)}")
    # reanálise do município
    rea = read_sisam(sisam_files(sisam_dir), municipios.pop())
    # observação: média horária entre estações, depois média diária com a regra das 18 h
    obs = daily_mean(obs_media_horaria(cetesb_dir, estacoes, min_estacoes), min_hours)
    # índice diário completo no período da reanálise
    dias = pd.date_range(rea.index.min(), rea.index.max(), freq="D", name="date")
    # reanálise no índice completo
    df = rea.reindex(dias)
    # observação como primeira coluna, como no benchmark
    df.insert(0, TARGET, obs.reindex(dias))
    # previsão média nas mesmas estações (trilha de previsão)
    if cams_forecast_dir is not None:
        # leitura no formato do benchmark (a pasta tem o nome da média, não um código)
        fc = read_cams_forecast(cams_forecast_dir, nome_previsao)  # type: ignore[arg-type]
        # colunas fc_* depois das rea_*
        df = df.join(fc.reindex(dias))
        # marca a trilha de previsão
        df.attrs["forecast"] = True
    # o runner exige um código inteiro; 0 indica estação virtual (sobrescrito no relatório)
    df.attrs["station"] = 0
    # devolve a tabela
    return df
