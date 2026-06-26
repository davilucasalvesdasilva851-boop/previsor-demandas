import pytest
import numpy as np

from previsor_demanda_corrigido import (
    ingenuo,
    media_movel,
    media_ponderada,
    suavizacao,
    regressao,
    calcular_mad,
    calcular_erro_historico,
    recomendacao
)

# ======================================================
# CASO 1 — DEMANDA ESTÁVEL
# ======================================================

def test_demanda_estavel_media_movel():

    dados = [100,100,100,100,100]

    resultado = media_movel(dados, janela=3)

    assert resultado == 100


# ======================================================
# CASO 2 — DEMANDA CRESCENTE
# ======================================================

def test_demanda_crescente_regressao():

    dados = [50,60,70,80,90]

    resultado = regressao(dados, semanas=1)

    assert resultado[0] > 90


# ======================================================
# CASO 3 — DEMANDA DECRESCENTE
# ======================================================

def test_demanda_decrescente_suavizacao():

    dados = [120,110,100,90,80]

    resultado = suavizacao(dados, alfa=0.3)

    assert resultado < 120


# ======================================================
# CASO 4 — DEMANDA IRREGULAR
# ======================================================

def test_demanda_irregular_media_ponderada():

    dados = [30,150,40,200,60]

    pesos = [0.1,0.2,0.2,0.2,0.3]

    resultado = media_ponderada(dados, pesos)

    assert resultado > 0
    assert isinstance(resultado, float)


# ======================================================
# CASO 5 — POUCOS DADOS
# ======================================================

def test_poucos_dados_regressao():

    dados = [100]

    resultado = regressao(dados, semanas=2)

    assert resultado == [100.0,100.0]


# ======================================================
# CASO 6 — DADOS COM ZERO
# ======================================================

def test_dados_com_zero():

    dados = [0,0,10,0,20]

    resultado = suavizacao(dados, alfa=0.3)

    assert resultado >= 0


# ======================================================
# CASO 7 — ENTRADA INVÁLIDA
# ======================================================

def test_entrada_invalida():

    entrada = "100,abc,200"

    with pytest.raises(ValueError):

        [float(x.strip()) for x in entrada.split(",")]


# ======================================================
# CASO 8 — COMPARAÇÃO ENTRE MÉTODOS
# ======================================================

def test_calculo_mad():

    reais = [100,110,120]

    previstos = [98,112,119]

    resultado = calcular_mad(reais, previstos)

    assert round(resultado,2) == 1.67


# ======================================================
# CASO 9 — PREVISÃO DE 4 SEMANAS
# ======================================================

def test_previsao_4_semanas():

    dados = [40,45,50,55,60]

    resultado = regressao(dados, semanas=4)

    assert len(resultado) == 4


# ======================================================
# CASO 10 — RECOMENDAÇÃO GERENCIAL
# ======================================================

def test_recomendacao_gerencial():

    historico = [100,120,140,160,180]

    futuro = [250]

    resultado = recomendacao(historico, futuro)

    assert any(
        "produção" in item.lower()
        for item in resultado
    )


# ======================================================
# TESTE EXTRA — ERRO HISTÓRICO
# ======================================================

def test_erro_historico_media_movel():

    dados = [10,20,30,40,50]

    erro = calcular_erro_historico(
        dados,
        metodo="Média Móvel",
        janela=2
    )

    assert erro is not None
    assert erro >= 0


# ======================================================
# TESTE EXTRA — MÉDIA PONDERADA SEM PESOS
# ======================================================

def test_media_ponderada_sem_pesos():

    dados = [10,20,30]

    resultado = media_ponderada(dados, [])

    assert resultado == np.mean(dados)
