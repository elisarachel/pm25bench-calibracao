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

# verificação das colunas de previsão
from pm25bench_calibracao.features import exigir_previsao

#: Valores aceitos de ``entrada``.
ENTRADAS = ("reanalise", "previsao")


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

    Com ``entrada="reanalise"`` (padrão), como não existe previsão da reanálise para t+k no
    contrato padrão, o mesmo valor (a calibração do dia t) é devolvido para todos os leads.

    Com ``entrada="previsao"`` (trilha de previsão do CAMS), a reta é ajustada nos pares
    (``fc_pm2_5_l0(d)``, ``obs(d)``) de d = t-n+1 a t-1, cada dia comparado com a previsão de
    lead 0 emitida nele mesmo, e aplicada a ``fc_pm2_5_l{k}(t)`` para cada lead k: um valor
    por lead. É a forma clássica de Model Output Statistics com janela móvel. A reta de lead
    0 é usada em todos os leads, supondo que o viés do CAMS não depende do lead; o erro que
    cresce com o lead não é corrigido.

    Parameters
    ----------
    min_pares : int
        Número mínimo de pares válidos na janela para ajustar a reta.
    var_min : float
        Variância amostral mínima da variável explicativa na janela para ajustar a reta.
    entrada : {"reanalise", "previsao"}
        Variável calibrada: a reanálise do SISAM ou a previsão do CAMS.
    """

    # declarado na classe para cumprir o protocolo Forecaster; o valor é definido no __init__
    name: str

    def __init__(
        self, min_pares: int = 3, var_min: float = 1e-6, entrada: str = "reanalise"
    ) -> None:
        # só duas entradas são aceitas; qualquer outra é erro de configuração
        if entrada not in ENTRADAS:
            raise ValueError(f"entrada deve ser 'reanalise' ou 'previsao', não {entrada!r}")
        # nome distinto por entrada para o runner não misturar os resultados
        self.name = "dom" if entrada == "reanalise" else "dom_previsao"
        # guarda a entrada escolhida
        self.entrada = entrada
        # coluna explicativa na janela: reanálise do dia ou previsão de lead 0 do dia
        self._x = "rea_pm2_5" if entrada == "reanalise" else "fc_pm2_5_l0"
        # guarda o número mínimo de pares para o ajuste
        self.min_pares = min_pares
        # guarda a variância mínima da reanálise para o ajuste
        self.var_min = var_min
        # fator de reserva; None indica que fit ainda não foi chamado
        self._escala: float | None = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        """Calcula o fator de escala de reserva no treino. A validação não é usada."""
        # na trilha de previsão, o treino precisa ter as colunas fc_*
        if self.entrada == "previsao":
            exigir_previsao(train, [self._x], self.name)
        # dias do treino em que observação e variável explicativa são ambas válidas
        ok = train[[TARGET, self._x]].dropna()
        # soma da variável explicativa nesses dias (denominador da razão)
        soma_rea = float(ok[self._x].sum())
        # razão das somas = média(obs) / média(x); 1.0 se o treino não tiver dados
        self._escala = float(ok[TARGET].sum()) / soma_rea if soma_rea > 0 else 1.0

    def predict(self, context: pd.DataFrame, leads: Sequence[int]) -> np.ndarray:
        """Calibra a entrada do dia t com a reta ajustada nos dias t-n+1 ... t-1."""
        # o protocolo exige fit antes de predict, como nos baselines do benchmark
        if self._escala is None:
            raise RuntimeError("chame fit antes de predict")
        # valores a calibrar, um por lead
        if self.entrada == "previsao":
            # colunas da previsão emitida em t para cada t + k
            cols = [f"fc_pm2_5_l{k}" for k in leads]
            # confere as colunas antes de usar
            exigir_previsao(context, sorted({self._x, *cols}), self.name)
            # previsões da linha t (emitidas em t), uma por lead
            x_t = context[cols].iloc[-1].to_numpy(dtype="float64")
        else:
            # a reanálise do dia t, repetida para todos os leads
            x_t = np.full(len(leads), float(context["rea_pm2_5"].iloc[-1]))
        # janela de treino: todas as linhas menos a última (a obs de t nunca é lida)
        janela = context[[self._x, TARGET]].iloc[:-1].dropna()
        # variável explicativa na janela
        x = janela[self._x].to_numpy(dtype="float64")
        # observação na janela (variável resposta)
        obs = janela[TARGET].to_numpy(dtype="float64")
        # a reta só é estimável com pares suficientes e x que varia na janela
        if len(x) >= self.min_pares and float(np.var(x, ddof=1)) > self.var_min:
            # polyfit de grau 1 devolve [b, a] (maior grau primeiro) pela solução OLS
            b, a = np.polyfit(x, obs, 1)
            # aplica a reta a cada valor do dia t
            valores = a + b * x_t
        else:
            # reserva: correção multiplicativa fixa estimada no treino
            valores = self._escala * x_t
        # concentração negativa não tem sentido físico
        return np.asarray(np.maximum(valores, 0.0), dtype="float64")


class DOMPrevisao(DOM):
    """``DOM(entrada="previsao")``, registrado como ``dom_previsao``."""

    def __init__(self, min_pares: int = 3, var_min: float = 1e-6) -> None:
        # mesma classe, com a previsão do CAMS como entrada
        super().__init__(min_pares=min_pares, var_min=var_min, entrada="previsao")
