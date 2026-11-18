"""pm25bench-calibracao: modelos de calibração da reanálise para o benchmark pm25bench.

- ``DOM``: correção linear local (OLS) ajustada na janela de contexto de cada amostra.
- ``GBM``: gradient boosting de árvores (LightGBM), um regressor por lead.
- ``DOMPrevisao`` e ``GBMPrevisao``: as mesmas ideias aplicadas à previsão do CAMS (trilha
  de previsão do pm25bench, colunas ``fc_*``).
"""

# reexporta as classes registradas como plugins
from pm25bench_calibracao.dom import DOM, DOMPrevisao
from pm25bench_calibracao.gbm import GBM, GBMPrevisao

# API pública do pacote
__all__ = ["DOM", "GBM", "DOMPrevisao", "GBMPrevisao"]
