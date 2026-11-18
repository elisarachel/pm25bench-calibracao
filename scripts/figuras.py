"""Figuras do trabalho final (experimento da média de 5 estações), em ``figuras/``.

Uso (da raiz do plugin, depois de ``scripts/rodar_media.py``)::

    ../Benchmark/.venv/bin/python scripts/figuras.py configs/benchmark_media_sp5_previsao.yaml

Cada figura responde a uma pergunta:

1. ``1_serie_mensal.png``: o viés da previsão do CAMS é estável no tempo?
2. ``2_bland_altman.png``: o erro do CAMS é aditivo ou multiplicativo?
3. ``3_ciclo_sazonal.png``: o viés depende da época do ano?
4. ``4_autocorrelacao.png``: por quantos dias a série depende de si mesma (blocos e gap)?
5. ``5_erro_por_lead.png``: quanto cada método ganha, e como o erro cresce com o horizonte?
6. ``6_diferenca_pareada.png``: o GBM é melhor que os outros métodos (hipótese)?
7. ``7_extremos.png``: os métodos subestimam os dias acima do P90?
8. ``8_serie_teste.png``: onde os métodos erram no período de teste?
"""

from __future__ import annotations

# argparse: leitura do caminho da config
import argparse

# Path: caminhos
from pathlib import Path

# matplotlib sem janela (gera arquivos)
import matplotlib

# backend de arquivo
matplotlib.use("Agg")

# pyplot: figuras
import matplotlib.pyplot as plt

# numpy: contas
import numpy as np

# pandas: tabelas
import pandas as pd

# configuração e divisão do benchmark
from pm25bench import load_config, split_by_time

# RMSE igual ao do runner
from pm25bench.metrics import rmse

# bootstrap pareado
from pm25bench_calibracao.avaliacao import tabela_diferencas

# tabela da média das estações
from pm25bench_calibracao.media import tabela_media

#: Pasta das figuras.
SAIDA = Path("figuras")

#: Data da troca de versão do sistema do CAMS que muda o nível da previsão.
TROCA_VERSAO = pd.Timestamp("2017-01-25")

#: Cor de cada modelo (paleta categórica validada, sempre a mesma cor para o mesmo modelo).
COR: dict[str, str] = {
    # regressão linear
    "linear_previsao": "#2a78d6",
    # previsão do CAMS sem correção
    "forecast_raw": "#eb6834",
    # LightGBM
    "gbm_previsao": "#1baf7a",
    # correção linear local
    "dom_previsao": "#eda100",
    # persistência
    "persistence": "#e87ba4",
    # climatologia
    "climatology": "#008300",
}

#: Marcador de cada modelo (codificação além da cor).
MARCA: dict[str, str] = {
    # um marcador distinto por modelo
    "linear_previsao": "o",
    "forecast_raw": "s",
    "gbm_previsao": "D",
    "dom_previsao": "^",
    "persistence": "v",
    "climatology": "P",
}

#: Nome de cada modelo nas figuras.
ROTULO: dict[str, str] = {
    # nomes curtos em português
    "linear_previsao": "regressão linear",
    "forecast_raw": "CAMS bruto",
    "gbm_previsao": "LightGBM",
    "dom_previsao": "DOM",
    "persistence": "persistência",
    "climatology": "climatologia",
}

#: Tinta principal (observação e textos).
TINTA = "#0b0b0b"

#: Tinta secundária (eixos e anotações).
TINTA2 = "#52514e"

#: Cinza dos eixos e da linha de referência.
CINZA = "#898781"

#: Linhas de grade.
GRADE = "#e1e0d9"


def estilo() -> None:
    """Estilo comum: fundo claro, grade fina, eixos discretos, fonte legível."""
    # parâmetros globais do matplotlib
    plt.rcParams.update(
        {
            # fundo da figura e dos eixos
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            # bordas: só esquerda e baixo
            "axes.spines.top": False,
            "axes.spines.right": False,
            # cor das bordas e dos ticks
            "axes.edgecolor": CINZA,
            "xtick.color": TINTA2,
            "ytick.color": TINTA2,
            # cor dos rótulos e títulos
            "axes.labelcolor": TINTA,
            "axes.titlecolor": TINTA,
            "text.color": TINTA,
            # grade fina e clara, sempre atrás dos dados
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRADE,
            "grid.linewidth": 0.6,
            # linhas finas
            "lines.linewidth": 1.6,
            # tamanhos de fonte
            "font.size": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "legend.frameon": False,
            # resolução dos arquivos
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def salvar(fig: plt.Figure, nome: str) -> None:
    """Grava a figura em ``figuras/<nome>`` e fecha."""
    # grava
    fig.savefig(SAIDA / nome)
    # libera a memória
    plt.close(fig)
    # registro
    print(f"figuras/{nome}")


def fig_serie_mensal(df: pd.DataFrame, fim_treino: pd.Timestamp, fim_val: pd.Timestamp) -> None:
    """1. Média mensal da observação e da previsão do CAMS (lead 0), 2015 em diante."""
    # período com previsão
    d = df.loc[df["fc_pm2_5_l0"].first_valid_index() :]
    # médias mensais
    m = d[["obs", "fc_pm2_5_l0"]].resample("MS").mean()
    # figura larga
    fig, ax = plt.subplots(figsize=(8, 3.2))
    # faixas dos períodos de validação e teste
    ax.axvspan(fim_treino, fim_val, color=GRADE, alpha=0.5, lw=0)
    # faixa do teste
    ax.axvspan(fim_val, m.index.max(), color=GRADE, alpha=0.9, lw=0)
    # observação
    ax.plot(m.index, m["obs"], color=TINTA, label="observado (média de 5 estações)")
    # previsão do CAMS
    ax.plot(m.index, m["fc_pm2_5_l0"], color=COR["forecast_raw"], label="CAMS bruto (lead 0)")
    # linha da troca de versão
    ax.axvline(TROCA_VERSAO, color=CINZA, lw=1, ls="--")
    # anotação da troca
    ax.annotate(
        "troca de versão\ndo CAMS (25/01/2017)",
        (TROCA_VERSAO, ax.get_ylim()[1] * 0.92),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=7,
        color=TINTA2,
        va="top",
    )
    # rótulos dos períodos, no rodapé (longe das curvas)
    for x, txt in [
        (pd.Timestamp("2015-02-01"), "treino"),
        (fim_treino, "val."),
        (fim_val, "teste"),
    ]:
        # texto no pé do gráfico, em coordenadas do eixo
        ax.text(
            x,
            0.02,
            f" {txt}",
            fontsize=7,
            color=TINTA2,
            transform=ax.get_xaxis_transform(),
            va="bottom",
        )
    # eixo y
    ax.set_ylabel("MP2,5 (µg/m³)")
    # título com a pergunta
    ax.set_title("O viés da previsão do CAMS é estável no tempo?", loc="left")
    # legenda abaixo do gráfico, fora da área de dados
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=2)
    # grava
    salvar(fig, "1_serie_mensal.png")


def fig_bland_altman(treino: pd.DataFrame) -> None:
    """2. Bland-Altman no treino: média (x) contra diferença CAMS − observação (y)."""
    # dias do treino com observação e previsão
    d = treino[["obs", "fc_pm2_5_l0"]].dropna()
    # eixo x: média das duas medidas
    x = (d["obs"] + d["fc_pm2_5_l0"]) / 2
    # eixo y: erro do CAMS
    y = d["fc_pm2_5_l0"] - d["obs"]
    # regime de cada dia (antes ou depois da troca de versão)
    depois = d.index >= TROCA_VERSAO
    # figura
    fig, ax = plt.subplots(figsize=(5.5, 4))
    # pontos de cada regime
    for mask, cor, rot in [
        (~depois, "#4a3aa7", "até 24/01/2017"),
        (depois, "#e34948", "a partir de 25/01/2017"),
    ]:
        # pontos pequenos e translúcidos
        ax.scatter(x[mask], y[mask], s=6, alpha=0.35, color=cor, label=rot, lw=0)
        # média do erro por faixa de x (mostra se o erro cresce com o nível)
        faixas = pd.qcut(x[mask], 10, duplicates="drop")
        # média de x e de y em cada faixa
        g = pd.DataFrame({"x": x[mask], "y": y[mask]}).groupby(faixas, observed=True).mean()
        # linha das médias por faixa
        ax.plot(g["x"], g["y"], color=cor, lw=2)
    # referência em zero (sem erro)
    ax.axhline(0, color=CINZA, lw=1)
    # rótulos
    ax.set_xlabel("(observado + CAMS) / 2  (µg/m³)")
    # eixo y
    ax.set_ylabel("CAMS − observado  (µg/m³)")
    # título com a pergunta
    ax.set_title("O erro do CAMS é aditivo ou multiplicativo? (treino)", loc="left")
    # legenda
    ax.legend(loc="upper left", markerscale=3)
    # grava
    salvar(fig, "2_bland_altman.png")


def fig_ciclo_sazonal(treino: pd.DataFrame) -> None:
    """3. Média por mês do ano no treino: observação e CAMS em cada regime."""
    # dias do treino com previsão
    d = treino.dropna(subset=["fc_pm2_5_l0"])
    # mês de cada dia
    mes = d.index.month
    # figura
    fig, ax = plt.subplots(figsize=(6, 3.4))
    # observação (todo o treino com previsão)
    ax.plot(
        range(1, 13), d["obs"].groupby(mes).mean(), color=TINTA, marker="o", ms=4, label="observado"
    )
    # CAMS em cada regime, com tracejados diferentes
    for mask, ls, rot in [
        (d.index < TROCA_VERSAO, ":", "CAMS até 24/01/2017"),
        (d.index >= TROCA_VERSAO, "-", "CAMS a partir de 25/01/2017"),
    ]:
        # média mensal da previsão no regime
        ax.plot(
            range(1, 13),
            d.loc[mask, "fc_pm2_5_l0"].groupby(mes[mask]).mean(),
            color=COR["forecast_raw"],
            ls=ls,
            marker="s",
            ms=4,
            label=rot,
        )
    # meses no eixo x
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    # eixo y
    ax.set_ylabel("MP2,5 (µg/m³)")
    # título com a pergunta
    ax.set_title("O viés do CAMS depende da época do ano? (treino)", loc="left")
    # legenda
    ax.legend()
    # grava
    salvar(fig, "3_ciclo_sazonal.png")


def fig_autocorrelacao(treino: pd.DataFrame) -> None:
    """4. Autocorrelação da observação diária (anomalia sazonal) no treino."""
    # observação diária
    obs = treino["obs"]
    # anomalia: tira a média de cada mês do ano (a sazonalidade inflaria a autocorrelação)
    anom = obs - obs.groupby(obs.index.month).transform("mean")
    # defasagens de 1 a 30 dias
    lags = np.arange(1, 31)
    # autocorrelação em cada defasagem (pares com NaN são ignorados)
    acf = [anom.autocorr(k) for k in lags]
    # faixa de 95% para ruído branco
    faixa = 1.96 / np.sqrt(anom.notna().sum())
    # figura
    fig, ax = plt.subplots(figsize=(6, 3.2))
    # barras finas
    ax.bar(lags, acf, width=0.6, color="#2a78d6", zorder=3)
    # faixa de significância (atrás das barras)
    ax.axhspan(-faixa, faixa, color=GRADE, lw=0, zorder=1)
    # rótulo da faixa
    ax.text(
        25, faixa, "faixa de 95% (ruído branco)", fontsize=7, color=TINTA2, ha="center", va="bottom"
    )
    # referência em zero
    ax.axhline(0, color=CINZA, lw=1)
    # marca o tamanho do bloco do bootstrap e do intervalo entre conjuntos
    ax.axvline(7, color=TINTA2, lw=1, ls="--")
    # anotação
    ax.text(
        7.3,
        max(acf) * 0.95,
        "7 dias (blocos do bootstrap\ne intervalo entre conjuntos)",
        fontsize=7,
        color=TINTA2,
        va="top",
    )
    # rótulos
    ax.set_xlabel("defasagem (dias)")
    # eixo y
    ax.set_ylabel("autocorrelação")
    # título com a pergunta
    ax.set_title("Por quantos dias a anomalia de MP2,5 depende de si mesma? (treino)", loc="left")
    # grava
    salvar(fig, "4_autocorrelacao.png")


def fig_erro_por_lead(scores: pd.DataFrame) -> None:
    """5. MAE por lead com IC 95% (bootstrap em blocos), um traço por modelo."""
    # figura
    fig, ax = plt.subplots(figsize=(6, 3.8))
    # ordem fixa dos modelos
    for m in [
        "forecast_raw",
        "climatology",
        "persistence",
        "dom_previsao",
        "linear_previsao",
        "gbm_previsao",
    ]:
        # linhas do modelo, em ordem de lead
        s = scores[scores["model"] == m].sort_values("lead")
        # linha com marcador
        ax.plot(s["lead"], s["mae"], color=COR[m], marker=MARCA[m], ms=6, label=ROTULO[m])
        # faixa do IC
        ax.fill_between(s["lead"], s["mae_lo"], s["mae_hi"], color=COR[m], alpha=0.12, lw=0)
    # leads inteiros
    ax.set_xticks([0, 1, 2, 3])
    # rótulos
    ax.set_xlabel("lead (dias)")
    # eixo y
    ax.set_ylabel("MAE (µg/m³)")
    # o eixo começa em zero (barra de erro absoluto)
    ax.set_ylim(bottom=0)
    # título com a pergunta
    ax.set_title("Quanto cada método ganha, e como o erro cresce com o horizonte?", loc="left")
    # legenda fora da área de dados
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # grava
    salvar(fig, "5_erro_por_lead.png")


def fig_diferenca_pareada(pred: pd.DataFrame, p90: float, cfg: object) -> pd.DataFrame:
    """6. RMSE(A) − RMSE(GBM) por lead, com IC 95% pareado, em todos os dias e acima do P90."""
    # modelos comparados com o GBM
    comparados = ["linear_previsao", "dom_previsao", "persistence"]
    # parâmetros do bootstrap iguais aos do benchmark
    args = {"block_days": cfg.block_days, "n_boot": cfg.n_boot, "seed": cfg.seed}  # type: ignore[attr-defined]
    # todos os dias
    todas = tabela_diferencas(pred, "gbm_previsao", comparados, rmse, **args)
    # só os dias acima do P90 do treino
    ext = tabela_diferencas(pred[pred["obs"] > p90], "gbm_previsao", comparados, rmse, **args)
    # um painel por comparação, mesmo eixo y
    fig, eixos = plt.subplots(1, 3, figsize=(9, 3.2), sharey=True)
    # percorre comparações
    for ax, m in zip(eixos, comparados, strict=True):
        # duas séries: todos os dias e extremos
        for tab, desl, marca, rot in [
            (todas, -0.08, "o", "todos os dias"),
            (ext, 0.08, "s", f"obs > P90 ({p90:.0f} µg/m³)"),
        ]:
            # linhas da comparação
            t = tab[tab["modelo"] == m]
            # ponto e barra do IC, na cor do modelo comparado
            ax.errorbar(
                t["lead"] + desl,
                t["dif"],
                yerr=[t["dif"] - t["ic_lo"], t["ic_hi"] - t["dif"]],
                fmt=marca,
                ms=5,
                capsize=3,
                color=COR[m],
                mfc=COR[m] if marca == "o" else "white",
                label=rot,
            )
        # referência em zero
        ax.axhline(0, color=CINZA, lw=1)
        # título do painel
        ax.set_title(f"{ROTULO[m]} − LightGBM", fontsize=9)
        # leads
        ax.set_xticks([0, 1, 2, 3])
        # eixo x
        ax.set_xlabel("lead (dias)")
    # eixo y
    eixos[0].set_ylabel("diferença de RMSE (µg/m³)\n> 0: LightGBM erra menos")
    # legenda
    eixos[0].legend(loc="upper left")
    # título geral com a pergunta
    fig.suptitle(
        "O LightGBM erra menos que os outros métodos? (IC 95%, bootstrap pareado)",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    # espaçamento
    fig.tight_layout()
    # grava
    salvar(fig, "6_diferenca_pareada.png")
    # devolve as tabelas juntas, para registro
    return pd.concat([todas.assign(dias="todos"), ext.assign(dias="obs>P90")])


def fig_extremos(pred: pd.DataFrame, p90: float, lead: int = 1) -> None:
    """7. Previsto contra observado nos dias acima do P90, um painel por método."""
    # dias extremos do lead escolhido
    d = pred[(pred["obs"] > p90) & (pred["lead"] == lead)]
    # métodos
    modelos = ["forecast_raw", "dom_previsao", "linear_previsao", "gbm_previsao"]
    # limites comuns
    lim = (0, float(max(d["obs"].max(), d["pred"].max())) * 1.05)
    # um painel por método
    fig, eixos = plt.subplots(1, 4, figsize=(10, 2.9), sharex=True, sharey=True)
    # percorre métodos
    for ax, m in zip(eixos, modelos, strict=True):
        # previsões do método
        s = d[d["model"] == m]
        # linha 1:1 (previsão perfeita)
        ax.plot(lim, lim, color=CINZA, lw=1, ls="--")
        # pontos
        ax.scatter(s["obs"], s["pred"], s=14, color=COR[m], marker=MARCA[m], lw=0, alpha=0.8)
        # viés médio no canto
        ax.text(
            0.04,
            0.96,
            f"viés {np.mean(s['pred'] - s['obs']):+.1f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color=TINTA2,
        )
        # título do painel
        ax.set_title(ROTULO[m], fontsize=9)
        # eixo x
        ax.set_xlabel("observado (µg/m³)")
        # limites iguais
        ax.set_xlim(lim)
        # limites iguais
        ax.set_ylim(lim)
    # eixo y
    eixos[0].set_ylabel("previsto (µg/m³)")
    # título geral com a pergunta
    fig.suptitle(
        f"Os métodos subestimam os dias acima do P90? (lead {lead}, n = {len(s)})",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    # espaçamento
    fig.tight_layout()
    # grava
    salvar(fig, "7_extremos.png")


def fig_serie_teste(pred: pd.DataFrame, lead: int = 1) -> None:
    """8. Série diária no teste: observado, regressão linear e LightGBM."""
    # previsões do lead escolhido
    d = pred[pred["lead"] == lead]
    # calendário diário do teste (dias sem amostra viram lacunas nas linhas)
    dias = pd.date_range(d["target_date"].min(), d["target_date"].max(), freq="D")
    # observação (igual para todos os modelos), indexada pela data alvo
    obs = d[d["model"] == "gbm_previsao"].set_index("target_date")["obs"].reindex(dias)
    # figura larga
    fig, ax = plt.subplots(figsize=(9, 3.2))
    # observação
    ax.plot(obs.index, obs, color=TINTA, lw=1.2, label="observado")
    # dois métodos
    for m in ["linear_previsao", "gbm_previsao"]:
        # série do método
        s = d[d["model"] == m].set_index("target_date")["pred"].reindex(dias)
        # linha fina
        ax.plot(s.index, s, color=COR[m], lw=1, label=ROTULO[m])
    # eixo y
    ax.set_ylabel("MP2,5 (µg/m³)")
    # título com a pergunta
    ax.set_title(
        f"Onde os métodos erram no período de teste? (lead {lead}; lacunas = dias "
        "sem amostra válida)",
        loc="left",
    )
    # legenda
    ax.legend(loc="upper left", ncols=3)
    # grava
    salvar(fig, "8_serie_teste.png")


def main() -> None:
    """Gera todas as figuras."""
    # analisador de argumentos
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # config do experimento
    p.add_argument("config", type=Path)
    # lê a linha de comando
    a = p.parse_args()
    # config
    cfg = load_config(a.config)
    # pasta das figuras
    SAIDA.mkdir(exist_ok=True)
    # estilo comum
    estilo()
    # tabela diária da média das estações (com a previsão)
    df = tabela_media(
        cfg.data.sisam_dir,
        cfg.data.cetesb_dir,
        cfg.data.cams_forecast_dir,  # type: ignore[arg-type]
        min_hours=cfg.data.min_hours,
    )
    # divisão do benchmark
    treino, _, _ = split_by_time(df, cfg.data.train_end, cfg.data.val_end, cfg.data.gap_days)
    # previsões e métricas gravadas pelo rodar_media.py
    pred = pd.read_csv(
        Path(cfg.output_dir) / "predictions.csv", parse_dates=["issue_date", "target_date"]
    )
    # métricas por modelo e lead
    scores = pd.read_csv(Path(cfg.output_dir) / "scores.csv")
    # P90 da observação no treino
    p90 = float(treino["obs"].dropna().quantile(0.90))
    # exploração
    fig_serie_mensal(df, pd.Timestamp(cfg.data.train_end), pd.Timestamp(cfg.data.val_end))
    # erro aditivo ou multiplicativo
    fig_bland_altman(treino)
    # sazonalidade
    fig_ciclo_sazonal(treino)
    # autocorrelação
    fig_autocorrelacao(treino)
    # resultados
    fig_erro_por_lead(scores)
    # hipótese
    dif = fig_diferenca_pareada(pred, p90, cfg)
    # registra a tabela da figura 6 junto dos resultados
    dif.round(3).to_csv(Path(cfg.output_dir) / "diferenca_pareada.csv", index=False)
    # extremos
    fig_extremos(pred, p90)
    # série do teste
    fig_serie_teste(pred)


# executa só quando chamado como script
if __name__ == "__main__":
    main()
