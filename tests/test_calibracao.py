"""Testes do plugin pm25bench-calibracao (dados sintéticos do pm25bench)."""

from __future__ import annotations

# numpy: comparações numéricas
import numpy as np

# pandas: construção de contextos à mão
import pandas as pd

# pytest: fixtures e verificação de exceções
import pytest

# API pública do benchmark
from pm25bench import Forecaster, load_model, make_samples, make_synthetic, split_by_time

# Sample: tipo das amostras; REANALYSIS_FEATURES: ordem das colunas rea_*
from pm25bench.data import REANALYSIS_FEATURES, Sample

# _features é privada; importada só para o teste de equivalência abaixo
from pm25bench.models.linear import _features

# modelos e funções de variáveis do plugin
from pm25bench_calibracao import DOM, GBM
from pm25bench_calibracao.features import (
    NOMES_BASE,
    NOMES_ESTENDIDAS,
    NOMES_PREVISAO,
    features_base,
    features_estendidas,
    features_previsao,
)

# leads do contrato atual
LEADS = [0, 1, 2, 3]

# GBM pequeno para os testes rodarem rápido (os padrões completos ficam para o benchmark)
GBM_RAPIDO = {"max_rounds": 200, "early_stopping": 20}


@pytest.fixture(scope="module")
def dados() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Treino, validação e teste sintéticos, com os mesmos cortes do template de plugin."""
    # tabela sintética no formato real, dividida em blocos de tempo
    return split_by_time(make_synthetic(), "2016-12-31", "2017-06-30")


@pytest.fixture(scope="module")
def amostras_teste(dados: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> list[Sample]:
    """Amostras do bloco de teste sintético."""
    # só o terceiro elemento (teste) interessa
    _, _, test = dados
    # todas as amostras válidas do teste
    return list(make_samples(test, 7, LEADS))


def _contexto(rea: list[float], obs: list[float]) -> pd.DataFrame:
    """Monta um contexto de 7 dias à mão: ``rea`` com 7 valores, ``obs`` com 6 (t fica NaN)."""
    # 7 datas consecutivas; a última é a data de emissão t
    idx = pd.date_range("2020-06-01", periods=7, freq="D", name="date")
    # demais colunas de reanálise com valores fixos (não afetam o DOM)
    ctx = pd.DataFrame({c: 1.0 for c in REANALYSIS_FEATURES}, index=idx)
    # reanálise de MP2,5 informada
    ctx["rea_pm2_5"] = rea
    # obs de t-6 a t-1 informada; NaN em t, como no benchmark
    ctx.insert(0, "obs", [*obs, np.nan])
    # devolve o contexto
    return ctx


# ---------------------------------------------------------------- 1. protocolo


@pytest.mark.parametrize("nome", ["dom", "gbm"])
def test_protocolo(
    nome: str,
    dados: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    amostras_teste: list[Sample],
) -> None:
    """Os modelos são encontrados pelo registro e cumprem o contrato de saída."""
    # argumentos reduzidos para o GBM; nenhum para o DOM
    kwargs = GBM_RAPIDO if nome == "gbm" else {}
    # carrega pelo entry point, como o runner faz
    model = load_model(nome, **kwargs)
    # tipagem estrutural: name, fit e predict compatíveis
    assert isinstance(model, Forecaster)
    # separa treino e validação
    train, val, _ = dados
    # ajusta sem ver o teste
    model.fit(train, val)
    # previsão em uma amostra de teste
    pred = model.predict(amostras_teste[0].context, LEADS)
    # um valor por lead
    assert pred.shape == (4,)
    # sem NaN nem infinito
    assert np.isfinite(pred).all()
    # sem concentração negativa
    assert (pred >= 0).all()


# ---------------------------------------------------------------- 2 e 3. DOM


def test_dom_recupera_a_reta(dados: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    """Com obs = 2 + 1,5·rea exato na janela, o DOM devolve 2 + 1,5·rea(t)."""
    # reanálise variando na janela; o último valor (t) é o calibrado
    rea = [10.0, 14.0, 9.0, 20.0, 16.0, 12.0, 18.0]
    # observação exatamente na reta nos 6 dias da janela
    obs = [2 + 1.5 * r for r in rea[:-1]]
    # modelo ajustado (o fit só define a escala de reserva)
    model = DOM()
    # usa o treino sintético
    model.fit(dados[0])
    # previsão para o contexto construído
    pred = model.predict(_contexto(rea, obs), LEADS)
    # todos os leads iguais à reta aplicada em rea(t)
    np.testing.assert_allclose(pred, 2 + 1.5 * rea[-1], rtol=0, atol=1e-9)


def test_dom_usa_reserva(dados: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    """Com rea_pm2_5 constante na janela, a reta é indefinida e vale a escala do treino."""
    # reanálise constante na janela e diferente em t
    rea = [15.0] * 6 + [22.0]
    # observação qualquer na janela
    obs = [10.0, 12.0, 9.0, 11.0, 13.0, 8.0]
    # modelo ajustado no treino sintético
    model = DOM()
    # define a escala de reserva
    model.fit(dados[0])
    # escala esperada: soma(obs) / soma(rea) nos dias válidos do treino
    ok = dados[0][["obs", "rea_pm2_5"]].dropna()
    # mesma conta do modelo
    escala = ok["obs"].sum() / ok["rea_pm2_5"].sum()
    # previsão para o contexto construído
    pred = model.predict(_contexto(rea, obs), LEADS)
    # reserva aplicada ao dia t
    np.testing.assert_allclose(pred, escala * rea[-1], rtol=1e-12)


def test_dom_exige_fit() -> None:
    """predict antes de fit levanta RuntimeError, como nos baselines."""
    # contexto qualquer
    ctx = _contexto([1.0] * 7, [1.0] * 6)
    # sem fit, o modelo não tem escala de reserva
    with pytest.raises(RuntimeError):
        DOM().predict(ctx, LEADS)


# ---------------------------------------------------------------- 4. variáveis


def test_features_base_igual_a_regressao_do_benchmark(amostras_teste: list[Sample]) -> None:
    """features_base reproduz pm25bench.models.linear._features sem o intercepto.

    _features é privada no benchmark e pode mudar sem aviso. Este teste existe para acusar
    essa mudança: se falhar, o GBM deixou de receber as mesmas entradas da linear_regression,
    e a comparação entre os dois perde o sentido.
    """
    # várias amostras espalhadas pelo teste
    for s in amostras_teste[::25]:
        # igualdade exata, elemento a elemento
        np.testing.assert_array_equal(features_base(s.context), _features(s.context)[1:])


def test_nomes_tem_o_tamanho_dos_vetores(amostras_teste: list[Sample]) -> None:
    """As listas de nomes acompanham os vetores (usadas em importancias)."""
    # contexto de uma amostra real
    ctx = amostras_teste[0].context
    # um nome por variável base
    assert len(NOMES_BASE) == len(features_base(ctx))
    # um nome por variável estendida
    assert len(NOMES_ESTENDIDAS) == len(features_estendidas(ctx))


# ---------------------------------------------------------------- 5. vazamento


@pytest.fixture(scope="module")
def modelos_ajustados(
    dados: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> list[DOM | GBM]:
    """DOM e as duas variantes do GBM, ajustados uma vez para os testes de vazamento."""
    # separa treino e validação
    train, val, _ = dados
    # os três modelos do plugin
    modelos: list[DOM | GBM] = [
        DOM(),
        GBM(**GBM_RAPIDO),
        GBM(variaveis="estendidas", **GBM_RAPIDO),
    ]
    # ajusta cada um
    for m in modelos:
        m.fit(train, val)
    # devolve os modelos prontos
    return modelos


def test_sem_vazamento_obs_do_dia_t(
    modelos_ajustados: list[DOM | GBM], amostras_teste: list[Sample]
) -> None:
    """Trocar a obs do dia t (última linha do contexto) não muda nenhuma previsão."""
    # amostra de teste qualquer
    s = amostras_teste[10]
    # cópia do contexto com um valor absurdo na obs de t
    alterado = s.context.copy()
    # última linha, coluna obs
    alterado.iloc[-1, alterado.columns.get_loc("obs")] = 999.0
    # cada modelo precisa dar exatamente o mesmo resultado
    for m in modelos_ajustados:
        np.testing.assert_array_equal(m.predict(s.context, LEADS), m.predict(alterado, LEADS))


def test_sem_vazamento_dias_futuros(
    dados: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    modelos_ajustados: list[DOM | GBM],
    amostras_teste: list[Sample],
) -> None:
    """Alterar a tabela a partir de t (obs de t e tudo depois) não muda a amostra emitida em t."""
    # bloco de teste original
    test = dados[2]
    # amostra de referência
    s = amostras_teste[10]
    # cópia da tabela para alterar
    mod = test.copy()
    # obs do próprio dia t, que o benchmark já esconde
    mod.loc[s.issue_date, "obs"] = 999.0
    # todas as colunas de todos os dias depois de t
    mod.loc[mod.index > s.issue_date, :] = 999.0
    # recria as amostras e pega a da mesma data de emissão
    nova = next(a for a in make_samples(mod, 7, LEADS) if a.issue_date == s.issue_date)
    # cada modelo precisa dar exatamente o mesmo resultado
    for m in modelos_ajustados:
        np.testing.assert_array_equal(m.predict(s.context, LEADS), m.predict(nova.context, LEADS))


# ---------------------------------------------------------------- 6 e 7. GBM


def test_gbm_reprodutivel(
    dados: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], amostras_teste: list[Sample]
) -> None:
    """Dois ajustes com a mesma semente dão previsões idênticas."""
    # separa treino e validação
    train, val, _ = dados
    # dois modelos com a mesma configuração e semente
    a, b = GBM(seed=7, **GBM_RAPIDO), GBM(seed=7, **GBM_RAPIDO)
    # ajusta o primeiro
    a.fit(train, val)
    # ajusta o segundo
    b.fit(train, val)
    # compara em várias amostras
    for s in amostras_teste[::50]:
        np.testing.assert_array_equal(a.predict(s.context, LEADS), b.predict(s.context, LEADS))


def test_gbm_estendidas_e_nomes(
    modelos_ajustados: list[DOM | GBM], amostras_teste: list[Sample]
) -> None:
    """A variante estendida treina, prevê e tem nome próprio; as importâncias saem por lead."""
    # o terceiro modelo do fixture é o GBM estendido
    estendido = modelos_ajustados[2]
    # nome distinto para o runner não misturar com o gbm base
    assert estendido.name == "gbm_estendidas"
    # o gbm base mantém o nome registrado
    assert modelos_ajustados[1].name == "gbm"
    # previsão finita com 4 valores
    pred = estendido.predict(amostras_teste[0].context, LEADS)
    # forma e valores válidos
    assert pred.shape == (4,) and np.isfinite(pred).all()
    # importâncias: uma linha por variável, uma coluna por lead
    assert isinstance(estendido, GBM)
    # tabela de importâncias
    imp = estendido.importancias()
    # dimensões esperadas
    assert imp.shape == (len(NOMES_ESTENDIDAS), len(LEADS))


def test_gbm_variante_invalida() -> None:
    """Variante desconhecida é erro de configuração."""
    # qualquer valor fora de {"base", "estendidas"}
    with pytest.raises(ValueError):
        GBM(variaveis="outra")


def test_gbm_sem_validacao_avisa(dados: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    """Sem validação, treina com max_rounds // 4 árvores e emite um aviso."""
    # modelo pequeno
    model = GBM(**GBM_RAPIDO)
    # fit sem validação deve avisar
    with pytest.warns(UserWarning):
        model.fit(dados[0], None)
    # número fixo de árvores em todos os leads
    assert set(model._iteracoes.values()) == {GBM_RAPIDO["max_rounds"] // 4}


def test_gbm_lead_sem_modelo_usa_o_maior(
    dados: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], amostras_teste: list[Sample]
) -> None:
    """Lead pedido sem modelo usa o do maior lead treinado."""
    # treina só os leads 0 e 1
    model = GBM(leads=(0, 1), **GBM_RAPIDO)
    # ajuste com validação
    model.fit(dados[0], dados[1])
    # contexto de teste
    ctx = amostras_teste[0].context
    # o lead 3 (sem modelo) deve igualar o lead 1 (maior treinado)
    pred = model.predict(ctx, [1, 3])
    # mesmo valor
    assert pred[0] == pred[1]


# ---------------------------------------------------------------- trilha de previsão do CAMS


@pytest.fixture(scope="module")
def dados_fc() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dados sintéticos com as colunas fc_pm2_5_l0 ... fc_pm2_5_l3, mesmos cortes."""
    # tabela sintética com previsão, dividida em blocos de tempo
    return split_by_time(make_synthetic(forecast=True), "2016-12-31", "2017-06-30")


@pytest.fixture(scope="module")
def modelos_previsao(
    dados_fc: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> list[DOM | GBM]:
    """dom_previsao e gbm_previsao carregados pelo registro e ajustados uma vez."""
    # separa treino e validação
    train, val, _ = dados_fc
    # carrega pelos entry points, como o runner faz
    modelos: list[DOM | GBM] = [
        load_model("dom_previsao"),  # type: ignore[list-item]
        load_model("gbm_previsao", **GBM_RAPIDO),  # type: ignore[list-item]
    ]
    # ajusta cada um
    for m in modelos:
        m.fit(train, val)
    # devolve os modelos prontos
    return modelos


def test_previsao_cumpre_o_protocolo(
    modelos_previsao: list[DOM | GBM],
    dados_fc: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """As variantes de previsão têm nome próprio e devolvem um valor válido por lead."""
    # amostras do teste com previsão
    amostras = list(make_samples(dados_fc[2], 7, LEADS))
    # nomes registrados, na ordem do fixture
    assert [m.name for m in modelos_previsao] == ["dom_previsao", "gbm_previsao"]
    # cada modelo
    for m in modelos_previsao:
        # cumpre o protocolo estrutural
        assert isinstance(m, Forecaster)
        # previsão em uma amostra
        pred = m.predict(amostras[0].context, LEADS)
        # forma, finitude e não negatividade
        assert pred.shape == (4,) and np.isfinite(pred).all() and (pred >= 0).all()


def test_dom_previsao_recupera_a_reta_por_lead(
    dados_fc: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Com obs = 2 + 1,5·fc_l0 na janela, cada lead recebe 2 + 1,5·fc_l{k}(t)."""
    # contexto base (a reanálise não é usada por esta variante)
    ctx = _contexto([10.0] * 7, [0.0] * 6)
    # previsão de lead 0 variando na janela
    ctx["fc_pm2_5_l0"] = [10.0, 14.0, 9.0, 20.0, 16.0, 12.0, 18.0]
    # previsões da linha t para os leads 1 a 3 (as outras linhas não são usadas)
    for k, v in zip([1, 2, 3], [21.0, 25.0, 30.0], strict=True):
        ctx[f"fc_pm2_5_l{k}"] = v
    # observação exatamente na reta nos dias da janela
    ctx.iloc[:-1, ctx.columns.get_loc("obs")] = 2 + 1.5 * ctx["fc_pm2_5_l0"].iloc[:-1]
    # modelo ajustado no treino com previsão
    model = DOM(entrada="previsao")
    # define a escala de reserva
    model.fit(dados_fc[0])
    # previsões da linha t, uma por lead
    fc_t = np.array([18.0, 21.0, 25.0, 30.0])
    # um valor por lead, cada um na reta
    np.testing.assert_allclose(model.predict(ctx, LEADS), 2 + 1.5 * fc_t, rtol=0, atol=1e-9)


@pytest.mark.parametrize("nome", ["dom_previsao", "gbm_previsao"])
def test_previsao_sem_colunas_falha(
    nome: str,
    dados: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Sem as colunas fc_*, o fit levanta KeyError com a dica de configuração."""
    # modelo pelo registro
    model = load_model(nome)
    # dados sintéticos SEM previsão
    train, val, _ = dados
    # mensagem explica o que falta configurar
    with pytest.raises(KeyError, match="cams_forecast_dir"):
        model.fit(train, val)


def test_features_previsao_sem_colunas_falha(amostras_teste: list[Sample]) -> None:
    """A função de variáveis também acusa a falta das colunas."""
    # contexto sem fc_*
    with pytest.raises(KeyError, match="cams_forecast_dir"):
        features_previsao(amostras_teste[0].context, 0)


def test_previsao_sem_vazamento(
    modelos_previsao: list[DOM | GBM],
    dados_fc: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Trocar a obs do dia t não muda a previsão das variantes de previsão."""
    # amostra de teste qualquer
    s = list(make_samples(dados_fc[2], 7, LEADS))[10]
    # cópia do contexto com um valor absurdo na obs de t
    alterado = s.context.copy()
    # última linha, coluna obs
    alterado.iloc[-1, alterado.columns.get_loc("obs")] = 999.0
    # cada modelo precisa dar exatamente o mesmo resultado
    for m in modelos_previsao:
        np.testing.assert_array_equal(m.predict(s.context, LEADS), m.predict(alterado, LEADS))


def test_features_previsao(amostras_teste: list[Sample]) -> None:
    """features_previsao = base + fc do dia alvo + erro recente de fc_l0."""
    # amostra com previsão
    df = make_synthetic(forecast=True)
    # primeira amostra completa
    s = next(make_samples(df, 7, LEADS))
    # vetor para o lead 2
    v = features_previsao(s.context, 2)
    # tamanho igual ao da lista de nomes
    assert len(v) == len(NOMES_PREVISAO)
    # começa com as variáveis base
    np.testing.assert_array_equal(v[: len(NOMES_BASE)], features_base(s.context))
    # previsão do dia alvo = fc_l2 da linha t
    assert v[-2] == s.context["fc_pm2_5_l2"].iloc[-1]
    # erro recente = média de obs - fc_l0 nas linhas antes de t
    ctx = s.context.iloc[:-1]
    # mesma conta, feita à parte
    assert v[-1] == pytest.approx(float((ctx["obs"] - ctx["fc_pm2_5_l0"]).mean()))


def test_dom_entrada_invalida() -> None:
    """Entrada desconhecida é erro de configuração."""
    # qualquer valor fora de {"reanalise", "previsao"}
    with pytest.raises(ValueError):
        DOM(entrada="outra")
