"""pm25bench-calibracao: modelos de calibração da reanálise para o benchmark pm25bench.

- ``DOM``: correção linear local (OLS) ajustada na janela de contexto de cada amostra.
- ``GBM``: gradient boosting de árvores (LightGBM), um regressor por lead.
- ``DOMPrevisao``, ``GBMPrevisao`` e ``LinearPrevisao``: calibração da previsão do CAMS
  (colunas ``fc_*``), só com informação disponível na manhã do dia t (sem reanálise).
"""

# reexporta as classes registradas como plugins
from pm25bench_calibracao.dom import DOM, DOMPrevisao
from pm25bench_calibracao.gbm import GBM, GBMPrevisao
from pm25bench_calibracao.linear import LinearPrevisao

# API pública do pacote
__all__ = ["DOM", "GBM", "DOMPrevisao", "GBMPrevisao", "LinearPrevisao"]
