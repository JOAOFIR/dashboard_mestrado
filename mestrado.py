import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.graph_objects as go
import folium
from folium.plugins import Search
# from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from pydtmc import MarkovChain, HiddenMarkovModel
from sklearn.model_selection import KFold
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.lines as mlines


# Configuração da página
st.set_page_config(
    page_title=None,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None,
)

# Menu lateral
with st.sidebar:
    selected = option_menu(
        menu_title="Análise Exploratória",
        options=["Mapa de Trajetórias", "Dados Consolidados", "Predições"],
    )

# Carregar os dados
t_origem = gpd.read_file('t_origem.geojson')
t_destino = gpd.read_file('t_destino.geojson')
traj_novo = gpd.read_file('traj_novo.geojson')
origens_df = pd.read_csv('origenstotal30.csv')
destinos_df = pd.read_csv('destinostotal30.csv')

para_centroides = pd.concat([origens_df, destinos_df])

# Criar mapa base
m = folium.Map(location=[para_centroides['lat'].mean(), para_centroides['lon'].mean()],
               zoom_start=12, tiles='cartodbpositron', attr='CartoDB')

def highlight_feature(feature):
    return {
        'fillColor': 'yellow',
        'color': 'yellow',
        'weight': 3,
        'fillOpacity': 0.5
    }

# Estilos para grids de origem e destino
estilo_origem = {
    'fillOpacity': 0.5,
    'color': '#008000',
    'weight': 1,
    'fillColor': '#00FF00'
}

estilo_destino = {
    'fillOpacity': 0.5,
    'color': '#FFA500',
    'weight': 1,
    'fillColor': '#FFD700'
}

# Adicionar geometrias de origem ao mapa
grid_origem = gpd.GeoDataFrame(t_origem[['VehId_x_x', 'grid_origem', 'day_x', 'period_x', 'Frequency', 'geometry']], geometry='geometry')
grid_origem_json = grid_origem.to_json()

o = folium.GeoJson(
    grid_origem_json,
    name='Grids de Origem',
    style_function=lambda x: estilo_origem,
    highlight_function=highlight_feature,
    tooltip=folium.GeoJsonTooltip(fields=['VehId_x_x', 'grid_origem', 'day_x', 'period_x', 'Frequency'])
).add_to(m)

# Adicionar geometrias de destino ao mapa
grid_destino = gpd.GeoDataFrame(t_destino[['VehId_x_x', 'grid_destino', 'Frequency', 'geometry']], geometry='geometry')
grid_destino_json = grid_destino.to_json()

d = folium.GeoJson(
    grid_destino_json,
    name='Grids de Destino',
    style_function=lambda x: estilo_destino,
    highlight_function=highlight_feature,
    tooltip=folium.GeoJsonTooltip(fields=['VehId_x_x', 'grid_destino', 'Frequency'])
).add_to(m)

# Adicionar subtrajetórias ao mapa
traj_novo[['start_t', 'end_t', 'grid_origem', 'grid_destino']] = traj_novo[['start_t', 'end_t', 'grid_origem', 'grid_destino']].astype(str)
subtraj_ = traj_novo[['Trip', 'start_t', 'end_t', 'Frequency', 'VehId_x_x', 'grid_origem', 'grid_destino', 'geometry']]
estilo_ = {'fillOpacity': 0, 'color': '#0000FF', 'weight': 0.3}

e = folium.GeoJson(subtraj_, name='SubTrajetórias Completas', style_function=lambda x: estilo_,
                   highlight_function=highlight_feature,
                   tooltip=folium.GeoJsonTooltip(fields=['VehId_x_x', 'Trip', 'start_t', 'end_t', 'grid_origem', 'grid_destino', 'Frequency'])).add_to(m)

# Adicionar search para grids de origem, destino e subtrajetórias
Search(
    layer=o,
    geom_type="Polygon",
    search_label="grid_origem",
    placeholder="Buscar Grid Origem",
    collapsed=True,
).add_to(m)

Search(
    layer=d,
    geom_type="Polygon",
    search_label="grid_destino",
    placeholder="Buscar Grid Destino",
    collapsed=True
).add_to(m)

Search(
    layer=e,
    geom_type="Line",
    search_label="VehId_x_x",
    placeholder="Buscar Veículo que vem do grid Origem",
    collapsed=True
).add_to(m)

Search(
    layer=e,
    geom_type="Line",
    search_label="Trip",
    placeholder="Buscar Trip",
    collapsed=True
).add_to(m)

# Controle de camadas
folium.LayerControl().add_to(m)

# Exibir o mapa
if selected == "Mapa de Trajetórias":
    st.header("Trajetórias", divider=True)
    st.components.v1.html(folium.Figure().add_child(m).render(), width=1080, height=540)
    st.caption("Fonte dos dados brutos: https://github.com/gsoh/VED.")
    st.caption("Fonte dos dados brutos: https://github.com/gsoh/VED.")

# Gráfico de Barras
if selected == "Dados Consolidados":
    st.header("Dados Consolidados")

    # Selecionando a coluna para análise
#     campo = st.selectbox("Selecione o campo para contar valores repetidos:", ['day_x', 'period_x'])
    st.caption("Fonte dos dados brutos: https://github.com/gsoh/VED.")
    df = pd.read_csv(r"experimento30122024v5.csv")
    st.dataframe(df)


# Exibindo a tabela de repetições para o campo 'grid_origem', apenas com duplicados
if selected == "Predições":
    st.header("Predições")
    st.caption("Fonte dos dados brutos: https://github.com/gsoh/VED.")
    df = pd.read_csv(r"experimento30122024v5.csv")
    df = df[(df['Frequency_geo'] >= 2)]

    df['tile_ID_x_x_x'] = df['tile_ID_x_x_x'].astype(str)
    df['dia_turno'] = df['day_x_x'].astype(str) + '_' + df['period_x_x'].astype(str)


    # Simulação de dados
    # Substitua esta parte pelos seus dados reais
    veiculos_data = df.groupby("VehId_x_x")["tile_ID_x_x_x"].apply(list).to_dict()

    # Função para calcular a matriz de transição
    def calculate_transition_matrix(states):
        unique_states = sorted(set(state for state in states if isinstance(state, str) and state.strip()))
        if not unique_states:
            return np.array([]), []

        state_to_index = {state: i for i, state in enumerate(unique_states)}
        transition_matrix = np.zeros((len(unique_states), len(unique_states)))

        for (current_state, next_state) in zip(states[:-1], states[1:]):
            if current_state in state_to_index and next_state in state_to_index:
                i = state_to_index[current_state]
                j = state_to_index[next_state]
                transition_matrix[i][j] += 1

        # Normalizar a matriz e tratar linhas com soma zero
        for i in range(transition_matrix.shape[0]):
            row_sum = transition_matrix[i].sum()
            if row_sum > 0:
                transition_matrix[i] /= row_sum
            else:
                transition_matrix[i] = 1 / len(unique_states)  # Distribuir uniformemente

        return transition_matrix, unique_states

    # Lista para armazenar resultados
    resultados = []

    # Processar cada veículo
    for veiculo, sequencia in veiculos_data.items():
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        acuracias = []

        # Loop para os folds do KFold
        for fold, (train_index, test_index) in enumerate(kf.split(sequencia)):
            X_train, X_test = np.array(sequencia)[train_index], np.array(sequencia)[test_index]

            # Criar matriz de transição
            try:
                matriz_transicao, estados = calculate_transition_matrix(X_train)
                if matriz_transicao.size == 0:  # Pular se não há estados válidos
                    acuracias.append(0)
                    continue

                mc = MarkovChain(matriz_transicao, states=estados)
            except ValueError as e:
                print(f"Erro ao criar a matriz de transição para o veículo {veiculo}: {e}")
                acuracias.append(0)
                continue

            # Predições e cálculo de acurácia
            predicoes = []
            for estado_inicial in X_test[:-1]:
                if estado_inicial in estados:
                    proximo_estado_predito = mc.predict(initial_state=estado_inicial, steps=1)
                    predicoes.append(proximo_estado_predito[0] if proximo_estado_predito else None)
                else:
                    predicoes.append(None)

            # Verificar alinhamento entre predições e teste
            if len(predicoes) != len(X_test[1:]):
                print(f"Erro de alinhamento: Veículo {veiculo}, Predições: {len(predicoes)}, Teste: {len(X_test[1:])}")
                acuracias.append(0)
                continue

            # Calcular acurácia
            acertos = sum(1 for p, r in zip(predicoes, X_test[1:]) if p == r and p is not None)
            total = len(X_test) - 1
            acuracia = acertos / total if total > 0 else 0
            acuracias.append(acuracia)

        # Registrar resultados para o veículo
        erro_padrao = np.std(acuracias) / np.sqrt(len(acuracias))
        for fold, acuracia in enumerate(acuracias):
            resultados.append({"Veiculo": veiculo, "Fold": fold + 1, "Acuracia": acuracia, "Erro_Padrao": erro_padrao})

    # Criar DataFrame final
    df_resultados = pd.DataFrame(resultados)

# Parte do HMM:

    df.rename({'VehId_x_x': 'VehId', 'tile_ID_x_x_x': 'origem', 'tile_ID_y_y_y': 'destino', 'Frequency_geo_day_period': 'Frequency', 'day_x_x':'day_x', 'period_x_x':'period_x'}, axis=1, inplace=True)

    # Função para calcular a matriz de emissão e transição
    def calcular_matriz_emissao_transicao(df):
        df = df.copy()  # Evita problemas de cópia
        df['observacao'] = df['day_x'].astype(str) + "_" + df['period_x'].astype(str)
        df['estado_oculto'] = df['origem'].astype(str)

        estados_unicos = sorted(df['estado_oculto'].unique())
        observacoes_unicas = sorted(df['observacao'].unique())

        estado_para_indice = {estado: i for i, estado in enumerate(estados_unicos)}
        observacao_para_indice = {obs: i for i, obs in enumerate(observacoes_unicas)}

        matriz_emissao = np.zeros((len(estados_unicos), len(observacoes_unicas)))
        matriz_transicao = np.zeros((len(estados_unicos), len(estados_unicos)))

        for estado, observacao in zip(df['estado_oculto'], df['observacao']):
            i = estado_para_indice[estado]
            j = observacao_para_indice[observacao]
            matriz_emissao[i, j] += 1

        soma_emissao = matriz_emissao.sum(axis=1, keepdims=True)
        soma_emissao[soma_emissao == 0] = 1  # Evita divisão por zero
        matriz_emissao = matriz_emissao / soma_emissao

        for estado_atual, estado_seguinte in zip(df['estado_oculto'][:-1], df['estado_oculto'][1:]):
            i = estado_para_indice[estado_atual]
            j = estado_para_indice[estado_seguinte]
            matriz_transicao[i, j] += 1

        soma_transicao = matriz_transicao.sum(axis=1, keepdims=True)

        # Normalizar a matriz de transição, garantindo que todas as linhas somem 1
        for k in range(len(soma_transicao)):
            if soma_transicao[k, 0] > 0:
                matriz_transicao[k, :] = matriz_transicao[k, :] / soma_transicao[k, 0]
            else:
                # Se a soma for zero, configurar uma distribuição uniforme
                matriz_transicao[k, :] = np.ones(len(soma_transicao)) / len(soma_transicao)

        return matriz_emissao, matriz_transicao, estados_unicos, observacoes_unicas

    # Iterar pelos veículos
    resultados = []

    for veiculo_id in df['VehId'].unique():  # Iterando pelos veículos únicos
        print(f"Processando veículo {veiculo_id}...")

        # Selecionar os dados do veículo
        veiculo_data = df[df['VehId'] == veiculo_id]

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        for fold, (train_index, test_index) in enumerate(kf.split(veiculo_data)):
            train_data, test_data = veiculo_data.iloc[train_index].copy(), veiculo_data.iloc[test_index].copy()

            # Criando colunas para treino
            train_data['observacao'] =  train_data['day_x'].astype(str) + "_" + train_data['period_x'].astype(str)
            train_data['estado_oculto'] = train_data['origem'].astype(str)

            # Criando colunas para teste
            test_data['observacao'] = test_data['day_x'].astype(str) + "_" + test_data['period_x'].astype(str)
            test_data['estado_oculto'] = test_data['origem'].astype(str)

            # Calculando as matrizes de emissão e transição para o conjunto de treino
            matriz_emissao_treino, matriz_transicao_treino, estados_treino, observacoes_treino = calcular_matriz_emissao_transicao(train_data)

            # Lidar com casos de apenas 1 estado
            if matriz_transicao_treino.shape[0] == 1:
                print(f"Apenas 1 estado no conjunto de treino ({estados_treino[0]}). Acurácia definida como 0.0%.")
                resultados.append({"Veiculo": veiculo_id, "Fold": fold + 1, "Acuracia": 0.0, "Erro_Padrao": 0.0})
                continue

            # Verificar validade das dimensões
            if matriz_transicao_treino.shape[0] != matriz_transicao_treino.shape[1]:
                print(f"Conjunto de treino inválido: matriz de transição não é quadrada. Ignorando este fold.")
                resultados.append({"Veiculo": veiculo_id, "Fold": fold + 1, "Acuracia": 0.0, "Erro_Padrao": 0.0})
                continue

            if matriz_emissao_treino.shape[0] != len(estados_treino) or matriz_emissao_treino.shape[1] != len(observacoes_treino):
                print("Dimensões da matriz de emissão não correspondem aos estados ou observações. Ignorando este fold.")
                resultados.append({"Veiculo": veiculo_id, "Fold": fold + 1, "Acuracia": 0.0, "Erro_Padrao": 0.0})
                continue

            # Criar o modelo HMM
            hmm = HiddenMarkovModel(p=matriz_transicao_treino, e=matriz_emissao_treino, states=estados_treino, symbols=observacoes_treino)

            # Preparando dados de teste
            X_test = test_data['observacao'].tolist()
            y_test = test_data['estado_oculto'].tolist()

            # Filtrar símbolos do conjunto de teste para garantir compatibilidade
            X_test_filtrado = [symbol for symbol in X_test if symbol in observacoes_treino]

            # Construir o vetor de probabilidades com base na distribuição de estados iniciais
            frequencia_estados_iniciais = test_data['estado_oculto'].value_counts(normalize=True).to_dict()
            y = np.array([frequencia_estados_iniciais.get(estado, 0) for estado in estados_treino])

            # Normalizar para garantir que a soma seja 1 (caso necessário)
            y = y / y.sum()

            # Predição
            if not X_test_filtrado:
                print("Nenhuma observação válida no conjunto de teste após filtrar.")
                acuracia = 0.0
            else:
                # Realizar predição
                y_pred = hmm.predict(prediction_type='viterbi', symbols=X_test_filtrado, initial_status=y, output_indices=False)

                # Verificar se a predição é válida
                if y_pred is None or y_pred[1] is None:
    #                print("Predição inválida. Acurácia definida como 0.0%.")
                    acuracia = 0.0
                else:
                    # Filtrar estados ocultos correspondentes às observações filtradas
                    indices_validos = [i for i, symbol in enumerate(X_test) if symbol in observacoes_treino]
                    y_test_filtrado = [y_test[i] for i in indices_validos]

                    # Calcular acurácia
                    acertos = sum(1 for p, r in zip(y_pred[1], y_test_filtrado) if p == r)
                    total = len(y_test_filtrado)
                    acuracia = acertos / total if total > 0 else 0.0

            erro_padrao = np.std([acuracia]) / np.sqrt(1)  # Como há apenas uma acurácia por fold, o erro padrão é zero
            resultados.append({"Veiculo": veiculo_id, "Fold": fold + 1, "Acuracia": acuracia, "Erro_Padrao": erro_padrao})

    # Criar DataFrame final
    df_resultadoshmm = pd.DataFrame(resultados)

# Final:
    resultados_markov = df_resultados
    resultados = pd.DataFrame(resultados)
    # Verificar se ambos os DataFrames têm a mesma estrutura e são compatíveis para o teste
    # assert len(resultados) == len(resultados_markov), "Os DataFrames devem ter o mesmo número de linhas"
    # assert "Acuracia" in resultados.columns and "Acuracia" in resultados_markov.columns, "Ambos os DataFrames devem conter a coluna 'Acuracia'"

    # Reshape dos dados em (30 veículos, 10 folds)
    acuracias_resultados = df_resultados["Acuracia"].values.reshape(30, 10)  # 30 veículos, 10 folds
    acuracias_markov = df_resultadoshmm["Acuracia"].values.reshape(30, 10)  # 30 veículos, 10 folds

    # Inicializar lista para armazenar os valores de p e médias
    p_values = []
    medias_resultados = [np.mean(acuracias_resultados[i]) for i in range(30)]
    medias_markov = [np.mean(acuracias_markov[i]) for i in range(30)]

    # Calcular o valor de p para cada veículo (comparação entre os dois conjuntos de dados)
    for i in range(30):  # Para cada veículo
        _, p_value = ttest_ind(acuracias_resultados[i], acuracias_markov[i])
        p_values.append(p_value)

    # Obter os 5 menores valores de p e os veículos correspondentes
    indices_menores_p = np.argsort(p_values)[:5]  # Indices dos 5 menores p-valores

    # Primeiro gráfico: Comparação de acurácias por veículo
    fig1 = go.Figure()

    indices = np.arange(1, 31)  # 30 veículos

    fig1.add_trace(go.Bar(x=indices, y=medias_resultados, name='Markov', marker_color='skyblue'))
    fig1.add_trace(go.Bar(x=indices, y=medias_markov, name='HMM', marker_color='orange'))

    # Configuração do primeiro gráfico
    fig1.update_layout(
        title='Comparação de Acurácias por Veículo',
        xaxis_title='Veículos',
        yaxis_title='Acurácia Média (%)',
        barmode='group'
    )

    # Legenda com os 5 menores valores de p
    for i in range(5):
        fig1.add_annotation(
            x=indices_menores_p[i] + 1,
            y=max(medias_resultados[indices_menores_p[i]], medias_markov[indices_menores_p[i]]) + 1,
            text=f"p = {p_values[indices_menores_p[i]]:.3e}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-20
        )

    # Segundo gráfico: Comparação de médias globais
    media_global_markov = np.mean(medias_markov)
    media_global_hmm = np.mean(medias_resultados)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=["Markov", "HMM"], y=[media_global_markov, media_global_hmm], marker_color=["skyblue", "orange"]))

    fig2.update_layout(
        title='Comparação de Acurácia Média Global',
        xaxis_title='Método',
        yaxis_title='Acurácia Média (%)'
    )

    # Terceiro gráfico: Veículos com os menores valores de p
    fig3 = go.Figure()

    for i in range(5):
        idx = indices_menores_p[i]
        fig3.add_trace(go.Bar(
            x=[f"Veículo {idx + 1}"],
            y=[medias_resultados[idx]],
            name=f"Markov Veículo {idx + 1}",
            marker_color='skyblue'
        ))
        fig3.add_trace(go.Bar(
            x=[f"Veículo {idx + 1}"],
            y=[medias_markov[idx]],
            name=f"HMM Veículo {idx + 1}",
            marker_color='orange'
        ))

    # Terceiro gráfico: Valores de p dos 5 menores veículos
    fig3 = go.Figure()

    for i in range(5):
        idx = indices_menores_p[i]
        fig3.add_trace(go.Bar(
            x=[f"Veículo {idx + 1}"],
            y=[p_values[idx]],
            name=f"Veículo {idx + 1}",
            marker_color='purple'
        ))

    fig3.update_layout(
        title='Valores de p dos 5 Veículos com Menores Valores de p',
        xaxis_title='Veículos',
        yaxis_title='Valor de p',
        barmode='group'
    )

    # Exibir gráficos no Streamlit
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

    # Terceiro gráfico: Quadro com os 5 menores valores de p usando Matplotlib
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.axis('off')

    # Legenda com os 5 menores valores de p
    legend_labels = [
        f"Veículo {indices_menores_p[i] + 1} - p = {p_values[indices_menores_p[i]]:.3e}"
        for i in range(5)
    ]

    # Criar setas vermelhas para a legenda
    legend_handles = [
        mlines.Line2D([0, 1], [0, 0], color='red', marker='>', markersize=10, label=legend_labels[i])
        for i in range(5)
    ]

    # Adicionar a legenda com as setas
    ax3.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(-0.3, 1), fontsize=10, title="5 menores valores de p", title_fontsize=12)

    # Ajustar o layout para evitar que a legenda sobreponha o gráfico
    plt.tight_layout()

    # Exibir o gráfico com a legenda de setas
#    st.subheader("5 Menores Valores de p")
    st.pyplot(fig3)







