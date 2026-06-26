import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ======================================
# CONFIG
# ======================================

st.set_page_config(
    page_title="Previsor de Demanda",
    page_icon="📈",
    layout="wide"
)

# ======================================
# DARK MODE
# ======================================

st.markdown("""
<style>

.stApp{
background:#080B13;
color:white;
}

section[data-testid="stSidebar"]{
background:#111827;
}

div[data-testid="metric-container"]{
background:#121B2E;
padding:18px;
border-radius:15px;
border:1px solid #1F2A44;
}

.stButton button{
width:100%;
height:55px;
background:#00B8D9;
color:white;
border-radius:12px;
}

</style>
""",unsafe_allow_html=True)

# ======================================
# FUNÇÕES
# ======================================

def ingenuo(dados):

    return float(dados[-1])


def media_movel(
    dados,
    janela=3
):

    if not dados:
        return 0.0

    if len(dados) < janela:
        return float(np.mean(dados))

    return float(np.mean(dados[-janela:]))


def media_ponderada(
    dados,
    pesos
):

    if not dados:
        return 0.0

    if not pesos:
        return float(np.mean(dados))

    n=min(len(dados), len(pesos))

    dados_recorte=np.array(dados[-n:], dtype=float)
    pesos_recorte=np.array(pesos[-n:], dtype=float)

    if pesos_recorte.sum() == 0:
        return float(np.mean(dados_recorte))

    pesos_recorte=pesos_recorte/pesos_recorte.sum()

    return float(np.average(
        dados_recorte,
        weights=pesos_recorte
    ))


def suavizacao(
    dados,
    alfa=0.3
):

    if not dados:
        return 0.0

    prev=float(dados[0])

    for v in dados[1:]:

        prev=(
            alfa*float(v)
            +(1-alfa)*prev
        )

    return float(prev)


def regressao(
    dados,
    semanas=1
):

    if not dados:
        return [0.0] * semanas

    if len(dados) < 2:
        return [float(dados[-1])] * semanas

    x=np.arange(
        len(dados)
    ).reshape(-1,1)

    y=np.array(
        dados,
        dtype=float
    )

    modelo=LinearRegression()

    modelo.fit(
        x,
        y
    )

    futuro=np.arange(
        len(dados),
        len(dados)+semanas
    ).reshape(-1,1)

    return [
        float(v)
        for v in modelo.predict(futuro)
    ]


def calcular_mad(
    reais,
    previstos
):

    if len(reais) != len(previstos):
        return None

    return float(np.mean(
        np.abs(
            np.array(reais)
            -
            np.array(previstos)
        )
    ))


def calcular_erro_historico(
    dados,
    metodo,
    janela=3,
    alfa=0.3
):

    if len(dados) < 2:
        return None

    if metodo == "Média Móvel":

        if len(dados) < janela + 1:
            return None

        reais=dados[janela:]
        previstos=[
            media_movel(dados[i-janela:i], janela)
            for i in range(janela, len(dados))
        ]

    elif metodo == "Suavização":

        reais=dados[1:]
        previstos=[]
        prev=float(dados[0])

        for valor in dados[1:]:
            previstos.append(prev)
            prev=alfa*float(valor) + (1-alfa)*prev

    elif metodo == "Regressão":

        reais=dados[1:]
        previstos=[]

        for i in range(1, len(dados)):
            x=np.arange(i).reshape(-1, 1)
            y=np.array(dados[:i], dtype=float)
            modelo=LinearRegression()
            modelo.fit(x, y)
            previstos.append(
                float(modelo.predict(np.array([[i]]))[0])
            )

    else:
        return None

    return calcular_mad(reais, previstos)


# ======================================
# RECOMENDAÇÃO
# ======================================

def recomendacao(
    historico,
    futuro
):

    media=np.mean(
        historico
    )

    prox=futuro[0]

    texto=[]

    if media == 0:
        variacao = 0
    else:
        variacao=(
            (prox-media)
            /
            media
        )*100

    if variacao>10:

        texto.append(
            "📈 Demanda crescendo"
        )

    elif variacao<-10:

        texto.append(
            "📉 Demanda caindo"
        )

    else:

        texto.append(
            "➖ Demanda estável"
        )

    if prox>media*1.15:

        texto.append(
            "🏭 Sugere aumento da produção"
        )

    if prox<media*0.8:

        texto.append(
            "📦 Risco de excesso de estoque"
        )

    if prox>historico[-1]*1.2:

        texto.append(
            "⚠️ Possível falta de produto"
        )

    if abs(variacao)>20:

        texto.append(
            "🔧 Revisar capacidade produtiva"
        )

    return texto


# ======================================
# SIDEBAR
# ======================================

with st.sidebar:

    st.title(
        "⚙️ Configurações"
    )

    produto=st.text_input(
        "Produto"
    )

    entrada=st.text_area(
        "Demandas históricas"
    )

    semanas=st.slider(
        "Semanas Futuras",
        1,
        12,
        4
    )

    metodo=st.selectbox(
        "Método principal",
        [
            "Ingênuo",
            "Média Móvel",
            "Média Ponderada",
            "Suavização",
            "Regressão"
        ]
    )

    with st.expander("Comparação entre métodos", expanded=False):
        comparar_metodos=st.checkbox(
            "Ativar comparação",
            value=False
        )

        metodos_comparacao=st.multiselect(
            "Métodos para comparar",
            [
                "Ingênuo",
                "Média Móvel",
                "Média Ponderada",
                "Suavização",
                "Regressão"
            ],
            default=[
                "Média Móvel",
                "Média Ponderada",
                "Suavização",
                "Regressão"
            ],
            help="Selecione um ou mais métodos para comparar o desempenho histórico."
        ) if comparar_metodos else []

        if comparar_metodos:
            st.caption("Os controles abaixo serão usados apenas na comparação.")

# ======================
# PARAMETROS
# ======================

janela=3
alfa=0.3
pesos=[]
compare_janela=3
compare_alfa=0.3
compare_pesos=[]

dados_entrada=[]

if entrada:
    try:
        dados_entrada=[float(x.strip()) for x in entrada.split(",") if x.strip()]
    except ValueError:
        dados_entrada=[]

if metodo=="Média Móvel":

    janela=st.sidebar.slider(
        "Janela",
        2,
        8,
        3
    )

if metodo=="Média Ponderada":

    quantidade=max(2, len(dados_entrada)) if dados_entrada else 3

    st.sidebar.caption(f"Pesos definidos para {quantidade} demanda(s) histórica(s).")

    temp=[]

    for i in range(
        quantidade
    ):

        p=st.sidebar.slider(
            f"Peso {i+1}",
            0.1,
            1.0,
            1.0 if i==0 else 0.5
        )

        temp.append(
            p
        )

    soma=sum(
        temp
    )

    pesos=[
        x/soma
        for x in temp
    ]

if metodo=="Suavização":

    alfa=st.sidebar.slider(
        "Alfa",
        0.1,
        1.0,
        0.3
    )

if comparar_metodos:

    with st.expander("⚙️ Parâmetros da comparação", expanded=True):
        for metodo_comp in metodos_comparacao:

            if metodo_comp == "Média Móvel":
                with st.expander("📊 Média Móvel", expanded=False):
                    compare_janela=st.slider(
                        "Janela (comparação)",
                        2,
                        8,
                        janela
                    )

            elif metodo_comp == "Média Ponderada":
                with st.expander("⚖️ Média Ponderada", expanded=False):
                    quantidade_comp=max(2, len(dados_entrada)) if dados_entrada else 3

                    st.caption(f"Pesos usados na comparação: {quantidade_comp} ponto(s).")

                    temp_comp=[]

                    for i in range(quantidade_comp):
                        p_comp=st.slider(
                            f"Peso {i+1} (comparação)",
                            0.1,
                            1.0,
                            1.0 if i==0 else 0.5
                        )
                        temp_comp.append(p_comp)

                    soma_comp=sum(temp_comp)
                    compare_pesos=[x/soma_comp for x in temp_comp]

            elif metodo_comp == "Suavização":
                with st.expander("🔄 Suavização", expanded=False):
                    compare_alfa=st.slider(
                        "Alfa (comparação)",
                        0.1,
                        1.0,
                        alfa
                    )

            else:
                with st.expander(f"ℹ️ {metodo_comp}", expanded=False):
                    st.caption("Este método não exige parâmetros adicionais.")

gerar=st.sidebar.button(
    "🚀 Gerar"
)

# ======================================
# EXECUÇÃO
# ======================================

if gerar:

    try:
        dados=[
            float(x.strip())
            for x in entrada.split(",")
            if x.strip()
        ]
    except ValueError:
        st.error("Os dados históricos devem conter apenas números separados por vírgula.")
        st.stop()

    if len(dados) < 3:
        st.error("Informe pelo menos 3 valores históricos para gerar a comparação.")
        st.stop()

    if metodo=="Ingênuo":

        previsao_selecionada=[
            ingenuo(dados)
        ]*semanas

    elif metodo=="Média Móvel":

        valor=media_movel(
            dados,
            janela
        )

        previsao_selecionada=[
            valor
        ]*semanas

    elif metodo=="Média Ponderada":

        valor=media_ponderada(
            dados,
            pesos
        )

        previsao_selecionada=[
            valor
        ]*semanas

    elif metodo=="Suavização":

        valor=suavizacao(
            dados,
            alfa
        )

        previsao_selecionada=[
            valor
        ]*semanas

    else:

        previsao_selecionada=(
            regressao(
                dados,
                semanas
            )
        )

    comparacoes=[]

    if comparar_metodos and metodos_comparacao:

        for nome in metodos_comparacao:

            if nome == "Ingênuo":
                previsao_metodo=[ingenuo(dados)]*semanas
            elif nome == "Média Móvel":
                previsao_metodo=[media_movel(dados, compare_janela)]*semanas
            elif nome == "Média Ponderada":
                pesos_validos = compare_pesos if compare_pesos else [1] * min(3, len(dados))
                previsao_metodo=[media_ponderada(dados, pesos_validos)]*semanas
            elif nome == "Suavização":
                previsao_metodo=[suavizacao(dados, compare_alfa)]*semanas
            else:
                previsao_metodo=regressao(dados, semanas)

            comparacoes.append(
                (
                    nome,
                    previsao_metodo,
                    calcular_erro_historico(dados, nome, janela=compare_janela, alfa=compare_alfa)
                )
            )

    tabela_comp=[
        {
            "Método": nome,
            "Previsão inicial": round(previsao[0], 2),
            "MAD histórico": round(erro, 2) if erro is not None else "Não disponível"
        }
        for nome, previsao, erro in comparacoes
    ]

    melhor_metodo=None

    erros_validos=[
        (nome, erro)
        for nome, _, erro in comparacoes
        if erro is not None
    ]

    if erros_validos:
        melhor_metodo=min(
            erros_validos,
            key=lambda item: item[1]
        )[0]

    st.title(
        "📈 Previsor de Demanda"
    )

    c1,c2,c3,c4=st.columns(
        4
    )

    c1.metric(
        "Última",
        round(
            dados[-1]
        )
    )

    c2.metric(
        "Média",
        round(
            np.mean(dados),
            1
        )
    )

    c3.metric(
        "Próxima",
        round(
            previsao_selecionada[0],
            1
        )
    )

    mad=calcular_mad(
        dados[1:],
        dados[:-1]
    )

    c4.metric(
        "MAD",
        round(
            mad,
            2
        )
    )

    if comparar_metodos and comparacoes:

        st.subheader(
            "🔍 Comparação entre métodos"
        )

        st.dataframe(
            pd.DataFrame(tabela_comp),
            use_container_width=True
        )

        if melhor_metodo:
            st.success(
                f"O método com menor erro histórico foi {melhor_metodo}."
            )
        else:
            st.info(
                "Não foi possível calcular o erro histórico com a quantidade de dados informada."
            )

        st.info(
            "⚠️ Um menor erro no histórico não garante a melhor previsão futura, porque o comportamento do mercado pode mudar."
        )

    # gráfico

    fig=go.Figure()

    fig.add_trace(
        go.Scatter(
            y=dados,
            mode="lines+markers",
            name="Histórico"
        )
    )

    cores={
        "Ingênuo": "#F43F5E",
        "Média Móvel": "#22C55E",
        "Média Ponderada": "#A78BFA",
        "Suavização": "#F59E0B",
        "Regressão": "#06B6D4"
    }

    series_grafico=[(metodo, previsao_selecionada)]

    if comparar_metodos and comparacoes:
        series_grafico.extend(
            [(nome, previsao) for nome, previsao, _ in comparacoes if nome != metodo]
        )

    for nome, previsao in series_grafico:
        fig.add_trace(
            go.Scatter(
                x=[len(dados)-1] + list(range(len(dados), len(dados)+len(previsao))),
                y=[dados[-1]] + list(previsao),
                mode="lines+markers",
                name=nome,
                line=dict(color=cores.get(nome, "#06B6D4"))
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "🧠 Recomendação Gerencial"
    )

    for i in recomendacao(
        dados,
        previsao_selecionada
    ):

        st.success(
            i
        )

    st.subheader(
        "📌 Resumo"
    )

    st.write(
        f"""
Produto: {produto}

Método selecionado: {metodo}

Previsão Inicial:
{round(previsao_selecionada[0],2)}
"""
    )