"""Roda o benchmark com a média das estações (``pm25bench_calibracao.media``).

Uso (da raiz do plugin, no ambiente do benchmark)::

    ../Benchmark/.venv/bin/python scripts/rodar_media.py configs/benchmark_media_sp5_previsao.yaml

A config é um YAML normal do pm25bench com ``source: local``: ``sisam_dir``, ``cetesb_dir``
e (opcional) ``cams_forecast_dir`` apontam para as pastas de dados; ``station`` é ignorado,
porque a observação é a média de ``ESTACOES_SP5``.
"""

from __future__ import annotations

# argparse: leitura do caminho da config
import argparse

# Path: caminho da config
from pathlib import Path

# API do benchmark
from pm25bench import load_config, run_benchmark, write_report

# conjunto de estações e montagem da tabela
from pm25bench_calibracao.media import ESTACOES_SP5, MIN_ESTACOES, tabela_media


def main() -> None:
    """Lê a config, monta a tabela da média, roda o benchmark e grava o relatório."""
    # analisador de argumentos
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # config YAML
    p.add_argument("config", type=Path)
    # lê a linha de comando
    a = p.parse_args()
    # config validada pelo benchmark
    cfg = load_config(a.config)
    # a tabela da média é montada a partir de pastas locais
    if cfg.data.sisam_dir is None or cfg.data.cetesb_dir is None:
        raise SystemExit("a config precisa de source: local com sisam_dir e cetesb_dir")
    # tabela diária com a média das estações
    df = tabela_media(
        cfg.data.sisam_dir,
        cfg.data.cetesb_dir,
        cfg.data.cams_forecast_dir,
        min_hours=cfg.data.min_hours,
    )
    # roda o benchmark com o mesmo protocolo de sempre
    resultado = run_benchmark(cfg, df)
    # identifica a estação virtual no metadata.json e no leaderboard
    resultado.metadata["station"] = (
        f"média de {len(ESTACOES_SP5)} estações ({', '.join(map(str, ESTACOES_SP5))}), "
        f"mínimo de {MIN_ESTACOES} por hora"
    )
    # grava CSVs, leaderboard, metadados e figuras
    quadro = write_report(resultado, cfg.output_dir)
    # mostra o leaderboard
    print(quadro.read_text(encoding="utf-8"))


# executa só quando chamado como script
if __name__ == "__main__":
    main()
