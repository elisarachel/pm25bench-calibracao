"""Regressão linear da calibração da previsão: mesmas entradas do ``gbm_previsao``."""

from __future__ import annotations

# Sequence: tipo genérico da lista de leads
from collections.abc import Sequence

# numpy: álgebra linear
import numpy as np

# pandas: tabelas de treino e contexto
import pandas as pd

# make_samples: gera amostras no mesmo formato do teste
from pm25bench import make_samples

# variáveis da calibração da previsão e a verificação das colunas
from pm25bench_calibracao.features import exigir_previsao, features_previsao


class LinearPrevisao:
    """Regressão ridge por lead com as variáveis de ``features_previsao``.

    É o par linear do ``gbm_previsao``: os dois recebem exatamente as mesmas entradas
    (previsão do CAMS para o dia alvo, erro recente da previsão, observações até t-1 e
    sazonalidade, sem reanálise). A diferença entre eles isola o efeito da não linearidade.

    Para cada lead, as variáveis são padronizadas com média e desvio do treino,
    ``z = (x - μ) / σ``, e os coeficientes minimizam
    ``Σ (y_i - β0 - Σ_j β_j z_ij)² + α Σ_j β_j²``. A penalização ridge (``α``) encolhe os
    coeficientes e estabiliza o ajuste quando as variáveis são correlacionadas (por exemplo,
    a previsão e a observação de t-1). A solução fechada é
    ``β = (ZᵀZ + αI')⁻¹ Zᵀy``, com ``I'`` igual à identidade sem penalizar o intercepto.
    É a mesma conta da ``linear_regression`` do benchmark.

    Parameters
    ----------
    n_context : int
        Dias de contexto para montar as amostras de treino.
    alpha : float
        Penalização ridge (0 = mínimos quadrados puros), igual à do benchmark por padrão.
    leads : sequence of int
        Leads para os quais uma regressão é ajustada.
    """

    # nome usado pelo runner para identificar os resultados
    name = "linear_previsao"

    def __init__(
        self, n_context: int = 7, alpha: float = 1.0, leads: Sequence[int] = (0, 1, 2, 3)
    ) -> None:
        # dias de contexto
        self.n_context = n_context
        # penalização ridge
        self.alpha = alpha
        # leads como tupla imutável
        self._leads: tuple[int, ...] = tuple(leads)
        # coeficientes por lead (vazio até o fit)
        self._coef: dict[int, np.ndarray] = {}
        # média e desvio das variáveis por lead, calculados no treino
        self._mu: dict[int, np.ndarray] = {}
        # desvio padrão por lead
        self._sd: dict[int, np.ndarray] = {}

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        """Ajusta uma regressão por lead no treino. A validação não é usada."""
        # colunas de previsão necessárias (lead 0 para o erro recente e cada lead treinado)
        cols = sorted({"fc_pm2_5_l0", *(f"fc_pm2_5_l{k}" for k in self._leads)})
        # erro claro se o treino não tiver as previsões
        exigir_previsao(train, cols, self.name)
        # amostras de treino com contexto e alvos completos
        amostras = list(make_samples(train, self.n_context, self._leads))
        # sem amostras não há o que ajustar
        if not amostras:
            raise ValueError("treino sem amostras válidas")
        # alvos, uma coluna por lead
        y = np.stack([s.target for s in amostras])
        # um ajuste por lead
        for j, lead in enumerate(self._leads):
            # matriz de variáveis deste lead
            x = np.stack([features_previsao(s.context, lead) for s in amostras])
            # média de cada variável no treino
            mu = x.mean(axis=0)
            # desvio de cada variável no treino (evita divisão por zero)
            sd = x.std(axis=0) + 1e-9
            # variáveis padronizadas com uma coluna de 1 para o intercepto
            z = np.column_stack([np.ones(len(x)), (x - mu) / sd])
            # matriz de penalização: α na diagonal, sem penalizar o intercepto
            reg = self.alpha * np.eye(z.shape[1])
            # intercepto livre
            reg[0, 0] = 0.0
            # solução fechada da ridge
            self._coef[lead] = np.linalg.solve(z.T @ z + reg, z.T @ y[:, j])
            # guarda a padronização deste lead
            self._mu[lead], self._sd[lead] = mu, sd

    def predict(self, context: pd.DataFrame, leads: Sequence[int]) -> np.ndarray:
        """Prevê o MP2,5 diário em t + lead, truncado em zero."""
        # o protocolo exige fit antes de predict
        if not self._coef:
            raise RuntimeError("chame fit antes de predict")
        # maior lead treinado, usado quando pedem um lead sem modelo
        maior = max(self._coef)
        # previsões na ordem dos leads pedidos
        saida = []
        # percorre os leads pedidos
        for lead in leads:
            # modelo do lead, ou o do maior lead treinado (como a linear_regression)
            k = lead if lead in self._coef else maior
            # variáveis do contexto para esse lead, padronizadas com os valores do treino
            z = np.concatenate([[1.0], (features_previsao(context, k) - self._mu[k]) / self._sd[k]])
            # combinação linear, truncada em zero
            saida.append(max(float(z @ self._coef[k]), 0.0))
        # devolve como array 1D de float64
        return np.array(saida, dtype="float64")
