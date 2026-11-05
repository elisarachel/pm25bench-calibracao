"""pm25bench-calibracao: modelos de calibração da reanálise para o benchmark pm25bench.

- ``DOM``: correção linear local (OLS) ajustada na janela de contexto de cada amostra.
- ``GBM``: gradient boosting de árvores (LightGBM), um regressor por lead.
"""

# reexporta as classes registradas como plugins
from pm25bench_calibracao.dom import DOM
from pm25bench_calibracao.gbm import GBM

# API pública do pacote
__all__ = ["DOM", "GBM"]
