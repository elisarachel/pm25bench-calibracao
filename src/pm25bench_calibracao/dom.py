"""DOM: correção linear local da reanálise ajustada numa janela móvel (Monteiro et al., 2013)."""

from __future__ import annotations

# Sequence: tipo genérico da lista de leads
from collections.abc import Sequence

# numpy: mínimos quadrados e vetores
import numpy as np

# pandas: tabelas de treino e contexto
import pandas as pd

# nome da coluna observada ("obs")
from pm25bench.data import TARGET


class DOM:
    """Regressão linear ``obs = a + b·rea`` ajustada nos dias de contexto de cada amostra.

    Para cada amostra, os pares (``rea_pm2_5``, ``obs``) dos dias t-n+1 a t-1 formam a
    janela de treino. A reta de mínimos quadrados ordinários (OLS) minimiza
    ``Σ (obs_i - a - b·rea_i)²``, cuja solução fechada é

    ``b = cov(rea, obs) / var(rea)`` e ``a = média(obs) - b·média(rea)``.

    A previsão é ``a + b·rea_pm2_5(t)``, truncada em zero. Como a janela anda junto com a
    data de emissão, o modelo se reajusta todos os dias usando só observações até t-1,
    como no sistema operacional.

    Quando a janela não permite estimar a reta (menos de ``min_pares`` pares válidos ou
    ``rea_pm2_5`` quase constante, o que deixa ``var(rea)`` perto de zero e ``b``
    indefinido), usa um fator de escala fixo calculado no treino:
    ``escala = Σ obs / Σ rea_pm2_5``.

    Limitação: com o contrato atual (``n_context = 7``), a janela tem no máximo 6 pares,
    menos da metade dos 14 dias usados no sistema operacional (``CALIB_NDAYS=14``). Com
    tão poucos pontos a inclinação ``b`` é instável. Usar 14 dias exigiria
    ``n_context = 15``, uma mudança de contrato do benchmark.

    Como não existe previsão da reanálise para t+k neste contrato, o mesmo valor (a
    calibração do dia t) é devolvido para todos os leads.

    Parameters
    ----------
    min_pares : int
        Número mínimo de pares válidos na janela para ajustar a reta.
    var_min : float
        Variância amostral mínima de ``rea_pm2_5`` na janela para ajustar a reta.
    """

    # nome usado pelo runner para identificar os resultados
    name = "dom"

    def __init__(self, min_pares: int = 3, var_min: float = 1e-6) -> None:
        # guarda o número mínimo de pares para o ajuste
        self.min_pares = min_pares
        # guarda a variância mínima da reanálise para o ajuste
        self.var_min = var_min
        # fator de reserva; None indica que fit ainda não foi chamado
        self._escala: float | None = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        """Calcula o fator de escala de reserva no treino. A validação não é usada."""
        # dias do treino em que observação e reanálise são ambas válidas
        ok = train[[TARGET, "rea_pm2_5"]].dropna()
        # soma da reanálise nesses dias (denominador da razão)
        soma_rea = float(ok["rea_pm2_5"].sum())
        # razão das somas = média(obs) / média(rea); 1.0 se o treino não tiver dados
        self._escala = float(ok[TARGET].sum()) / soma_rea if soma_rea > 0 else 1.0

    def predict(self, context: pd.DataFrame, leads: Sequence[int]) -> np.ndarray:
        """Calibra a reanálise do dia t com a reta ajustada nos dias t-n+1 ... t-1."""
        # o protocolo exige fit antes de predict, como nos baselines do benchmark
        if self._escala is None:
            raise RuntimeError("chame fit antes de predict")
        # janela de treino: todas as linhas menos a última (a obs de t nunca é lida)
        janela = context[["rea_pm2_5", TARGET]].iloc[:-1].dropna()
        # reanálise na janela (variável explicativa)
        rea = janela["rea_pm2_5"].to_numpy(dtype="float64")
        # observação na janela (variável resposta)
        obs = janela[TARGET].to_numpy(dtype="float64")
        # reanálise do dia t, o valor a calibrar
        rea_t = float(context["rea_pm2_5"].iloc[-1])
        # a reta só é estimável com pares suficientes e reanálise que varia na janela
        if len(rea) >= self.min_pares and float(np.var(rea, ddof=1)) > self.var_min:
            # polyfit de grau 1 devolve [b, a] (maior grau primeiro) pela solução OLS
            b, a = np.polyfit(rea, obs, 1)
            # aplica a reta ao dia t
            valor = float(a + b * rea_t)
        else:
            # reserva: correção multiplicativa fixa estimada no treino
            valor = self._escala * rea_t
        # concentração negativa não tem sentido físico
        valor = max(valor, 0.0)
        # mesma calibração para todos os leads
        return np.full(len(leads), valor)
