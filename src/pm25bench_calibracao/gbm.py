"""GBM: gradient boosting de árvores (LightGBM), um regressor por lead."""

from __future__ import annotations

# warnings: aviso quando não há validação para a parada antecipada
import warnings

# Callable: tipo da função de variáveis; Sequence: tipo da lista de leads
from collections.abc import Callable, Sequence

# Any: tipo dos argumentos repassados em GBMPrevisao
from typing import Any

# LightGBM: implementação do gradient boosting de árvores
import lightgbm as lgb

# numpy: matrizes de variáveis e previsões
import numpy as np

# pandas: tabelas de treino e a saída de importancias()
import pandas as pd

# make_samples: gera amostras no mesmo formato do teste
from pm25bench import make_samples

# funções de variáveis e seus nomes
from pm25bench_calibracao.features import (
    NOMES_BASE,
    NOMES_ESTENDIDAS,
    NOMES_PREVISAO,
    exigir_previsao,
    features_base,
    features_estendidas,
    features_previsao,
)

#: Tipo das funções de variáveis: (contexto, lead) -> vetor.
FuncaoVariaveis = Callable[[pd.DataFrame, int], np.ndarray]

#: Função de variáveis e lista de nomes para cada valor aceito de ``variaveis``.
_VARIANTES: dict[str, tuple[FuncaoVariaveis, list[str]]] = {
    # mesmas entradas da linear_regression do benchmark (iguais em todos os leads)
    "base": (lambda ctx, lead: features_base(ctx), NOMES_BASE),
    # entradas base mais defasagens e viés recente (iguais em todos os leads)
    "estendidas": (lambda ctx, lead: features_estendidas(ctx), NOMES_ESTENDIDAS),
    # entradas base mais a previsão do CAMS para o dia alvo (dependem do lead)
    "previsao": (features_previsao, NOMES_PREVISAO),
}


class GBM:
    """Gradient boosting de árvores de regressão (LightGBM), um modelo por lead.

    O boosting soma ``M`` árvores pequenas, ``F_M(x) = Σ_m η·h_m(x)``. Cada árvore
    ``h_m`` é ajustada ao gradiente negativo da perda em relação à previsão acumulada
    ``F_{m-1}``. Com a perda quadrática (L2), ``L = (y - F)²/2``, o gradiente negativo é o
    próprio resíduo ``y - F_{m-1}(x)``: cada árvore nova aprende o que as anteriores ainda
    erram. A taxa de aprendizado ``η`` encolhe cada passo e torna o ajuste mais gradual.

    O número de árvores ``M`` é o principal controle de sobreajuste. Ele é escolhido por
    parada antecipada: o treino para quando o erro no conjunto de validação deixa de cair
    por ``early_stopping`` rodadas seguidas, e fica a iteração de menor erro de validação
    (``best_iteration``). A validação serve só para isso, e nunca entra no ajuste das
    árvores.

    O modelo final **não** é retreinado em treino + validação. Retreinar exigiria reutilizar
    o ``best_iteration`` encontrado com menos dados num conjunto maior, o que não é
    garantidamente equivalente, e misturaria no ajuste o período 2020-2021 (pandemia), que
    tem um regime de emissões atípico. Mantendo a validação fora do ajuste, o número de
    árvores escolhido corresponde exatamente ao modelo avaliado no teste.

    Parameters
    ----------
    n_context : int
        Dias de contexto para montar as amostras de treino (igual ao do benchmark).
    leads : sequence of int
        Leads para os quais um modelo é treinado.
    variaveis : {"base", "estendidas", "previsao"}
        Conjunto de variáveis de entrada (ver ``pm25bench_calibracao.features``). Com
        ``"previsao"`` (trilha de previsão do CAMS), o modelo de cada lead k recebe a
        previsão emitida em t para t + k, ``features_previsao(context, k)``.
    seed : int
        Semente do LightGBM, para resultados reprodutíveis.
    learning_rate : float
        Taxa de aprendizado ``η``.
    num_leaves : int
        Número máximo de folhas por árvore (complexidade de cada árvore).
    min_data_in_leaf : int
        Mínimo de amostras por folha (evita folhas que decoram poucos dias).
    max_rounds : int
        Número máximo de árvores.
    early_stopping : int
        Rodadas sem melhora na validação até parar.
    """

    # declarado na classe para cumprir o protocolo Forecaster; o valor é definido no __init__
    name: str

    def __init__(
        self,
        n_context: int = 7,
        leads: Sequence[int] = (0, 1, 2, 3),
        variaveis: str = "base",
        seed: int = 42,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        min_data_in_leaf: int = 20,
        max_rounds: int = 2000,
        early_stopping: int = 100,
    ) -> None:
        # só duas variantes são aceitas; qualquer outra é erro de configuração
        if variaveis not in _VARIANTES:
            raise ValueError(
                f"variaveis deve ser 'base', 'estendidas' ou 'previsao', não {variaveis!r}"
            )
        # nome distinto por variante para o runner não misturar os resultados
        self.name = "gbm" if variaveis == "base" else f"gbm_{variaveis}"
        # guarda os argumentos do construtor
        self.n_context = n_context
        # leads como tupla imutável
        self._leads: tuple[int, ...] = tuple(leads)
        # nome da variante escolhida
        self.variaveis = variaveis
        # função que transforma um contexto em vetor e os nomes das variáveis
        self._features, self._nomes = _VARIANTES[variaveis]
        # semente de aleatoriedade
        self.seed = seed
        # número máximo de árvores
        self.max_rounds = max_rounds
        # paciência da parada antecipada
        self.early_stopping = early_stopping
        # parâmetros passados ao LightGBM
        self._params: dict[str, object] = {
            # regressão com perda quadrática (L2)
            "objective": "regression",
            # métrica monitorada na validação: L2 (erro quadrático médio)
            "metric": "l2",
            # taxa de aprendizado η
            "learning_rate": learning_rate,
            # folhas por árvore
            "num_leaves": num_leaves,
            # mínimo de amostras por folha
            "min_data_in_leaf": min_data_in_leaf,
            # semente única que alimenta as demais sementes internas
            "seed": seed,
            # resultados idênticos entre execuções com a mesma semente
            "deterministic": True,
            # organização dos dados por linha, necessária para o determinismo
            "force_row_wise": True,
            # silencia as mensagens do LightGBM
            "verbose": -1,
        }
        # um Booster (modelo treinado) por lead
        self._modelos: dict[int, lgb.Booster] = {}
        # número de árvores escolhido para cada lead
        self._iteracoes: dict[int, int] = {}

    def _matrizes(self, df: pd.DataFrame) -> tuple[list[np.ndarray], np.ndarray]:
        """Monta ``X`` de cada lead (lista) e ``y`` (uma coluna por lead) de uma tabela diária."""
        # na trilha de previsão, a tabela precisa ter as colunas fc_*
        if self.variaveis == "previsao":
            # colunas de lead 0 (erro recente) e de cada lead treinado
            cols = sorted({"fc_pm2_5_l0", *(f"fc_pm2_5_l{k}" for k in self._leads)})
            # erro claro antes de gerar amostras
            exigir_previsao(df, cols, self.name)
        # amostras com contexto e alvos completos, como no teste
        amostras = list(make_samples(df, self.n_context, self._leads))
        # sem amostras, devolve matrizes vazias com o número certo de colunas
        if not amostras:
            # uma matriz vazia por lead
            vazias = [np.empty((0, len(self._nomes))) for _ in self._leads]
            # devolve as vazias e um y vazio
            return vazias, np.empty((0, len(self._leads)))
        # uma matriz de variáveis por lead (iguais entre leads nas variantes base/estendidas)
        xs = [np.stack([self._features(s.context, lead) for s in amostras]) for lead in self._leads]
        # uma linha de alvos (um por lead) por amostra
        y = np.stack([s.target for s in amostras])
        # devolve as matrizes por lead e os alvos
        return xs, y

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        """Treina um modelo por lead; a validação serve só para a parada antecipada."""
        # matrizes de treino
        xs_tr, y_tr = self._matrizes(train)
        # sem amostras de treino não há o que ajustar
        if len(y_tr) == 0:
            raise ValueError("treino sem amostras válidas")
        # matrizes de validação, ou nada se não houver validação
        xs_va, y_va = self._matrizes(val) if val is not None else ([], np.empty((0, 0)))
        # a parada antecipada só é possível com amostras de validação
        usa_validacao = len(y_va) > 0
        # sem validação, avisa que o número de árvores será fixo
        if not usa_validacao:
            warnings.warn(
                f"{self.name}: sem amostras de validação; treinando com "
                f"{self.max_rounds // 4} árvores fixas, sem parada antecipada",
                stacklevel=2,
            )
        # um modelo independente por lead
        for j, lead in enumerate(self._leads):
            # conjunto de treino do LightGBM para este lead (coluna j de y)
            d_tr = lgb.Dataset(xs_tr[j], label=y_tr[:, j], feature_name=self._nomes)
            # caminho com validação e parada antecipada
            if usa_validacao:
                # validação referenciada ao treino (usa os mesmos limites de histograma)
                d_va = lgb.Dataset(xs_va[j], label=y_va[:, j], reference=d_tr)
                # treina até max_rounds, parando se a validação não melhorar
                modelo = lgb.train(
                    self._params,
                    d_tr,
                    num_boost_round=self.max_rounds,
                    valid_sets=[d_va],
                    callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)],
                )
                # iteração de menor erro na validação
                iteracoes = int(modelo.best_iteration)
            else:
                # sem validação: número fixo e conservador de árvores
                iteracoes = self.max_rounds // 4
                # treina sem monitorar validação
                modelo = lgb.train(self._params, d_tr, num_boost_round=iteracoes)
            # guarda o modelo deste lead
            self._modelos[lead] = modelo
            # guarda o número de árvores usado na previsão
            self._iteracoes[lead] = iteracoes

    def predict(self, context: pd.DataFrame, leads: Sequence[int]) -> np.ndarray:
        """Prevê o MP2,5 diário em t + lead com o modelo de cada lead, truncado em zero."""
        # o protocolo exige fit antes de predict
        if not self._modelos:
            raise RuntimeError("chame fit antes de predict")
        # maior lead treinado, usado quando pedem um lead sem modelo
        maior = max(self._modelos)
        # previsões na ordem dos leads pedidos
        saida = []
        # percorre os leads pedidos
        for lead in leads:
            # modelo do lead, ou o do maior lead treinado (como a linear_regression)
            chave = lead if lead in self._modelos else maior
            # vetor de variáveis do contexto para o lead do modelo, como matriz de uma linha
            x = self._features(context, chave).reshape(1, -1)
            # previsão usando só as árvores até a melhor iteração (array de 1 elemento)
            prev = self._modelos[chave].predict(x, num_iteration=self._iteracoes[chave])
            # extrai o único valor como float
            valor = float(np.asarray(prev)[0])
            # trunca em zero: concentração negativa não tem sentido físico
            saida.append(max(valor, 0.0))
        # devolve como array 1D de float64
        return np.array(saida, dtype="float64")

    def importancias(self) -> pd.DataFrame:
        """Importância por ganho de cada variável em cada lead.

        O ganho de uma variável é a redução total da perda (soma dos quadrados dos resíduos)
        obtida em todas as divisões das árvores que usam essa variável. Valores maiores
        indicam variáveis que mais reduziram o erro no treino.

        Returns
        -------
        pandas.DataFrame
            Índice = nome da variável, uma coluna por lead.
        """
        # só faz sentido depois do treino
        if not self._modelos:
            raise RuntimeError("chame fit antes de importancias")
        # dicionário lead -> vetor de ganhos, usando só as árvores até a melhor iteração
        ganhos = {}
        # percorre os modelos treinados
        for lead, modelo in self._modelos.items():
            # número de árvores usado na previsão deste lead
            n = self._iteracoes[lead]
            # ganho total de cada variável nas árvores 1..n
            ganhos[lead] = modelo.feature_importance(importance_type="gain", iteration=n)
        # tabela com as variáveis nas linhas e os leads nas colunas
        tabela = pd.DataFrame(ganhos, index=pd.Index(self._nomes, name="variavel"))
        # nomeia o eixo das colunas
        tabela.columns.name = "lead"
        # devolve a tabela
        return tabela


class GBMPrevisao(GBM):
    """``GBM(variaveis="previsao")``, registrado como ``gbm_previsao``."""

    def __init__(self, **kwargs: Any) -> None:
        # não deixa trocar a variante por kwargs
        kwargs.pop("variaveis", None)
        # mesma classe, com as variáveis de previsão do CAMS
        super().__init__(variaveis="previsao", **kwargs)
