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
from sklearn.metrics import precision_score


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
        options=["Trajetórias para Cadeias de Markov", "Dados Consolidados", "Predições"],
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
if selected == "Trajetórias para Cadeias de Markov":
    st.header("Trajetórias - Markov", divider=True)
    st.components.v1.html(folium.Figure().add_child(m).render(), width=1080, height=540)
    st.caption("Fonte dos dados brutos: https://github.com/gsoh/VED.")

# Gráfico de Barras
if selected == "Dados Consolidados":
    st.header("Dados Consolidados")

    # Selecionando a coluna para análise
#     campo = st.selectbox("Selecione o campo para contar valores repetidos:", ['day_x', 'period_x'])
    st.caption("Fonte dos dados brutos: https://github.com/gsoh/VED.")
    df = pd.read_csv(r"result_final12.csv")
    st.dataframe(df)


# Exibindo a tabela de repetições para o campo 'grid_origem', apenas com duplicados
if selected == "Predições":
    st.header("Predições")
    st.caption("Fonte dos dados brutos: https://github.com/gsoh/VED.")
    df = pd.read_csv(r"result_final12.csv")
#    df = df[(df['Quantidade de Rotas Contadas'] >= 2)]

    df['grid_origem'] = df['grid_origem'].astype(str)
#    df['dia_turno'] = df['day_x_x'].astype(str) + '_' + df['period_x_x'].astype(str)

    filter = (df[df['Quantidade de Rotas Contadas'] >= 2])
    filter = (filter[filter['Quantidade de Rotas Contadas - Turno'] >= 2])
    df = filter.loc[filter.index.repeat(filter['Quantidade de Rotas Contadas'])].reset_index(drop=True)
    # Simulação de dados
    # Substitua esta parte pelos seus dados reais
    veiculos_data = df.groupby("VehId")["grid_origem"].apply(list).to_dict()

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
    resultados1 = []

    # Processar cada veículo
    for veiculo, sequencia in veiculos_data.items():
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        precisao_scores = []

        # Loop para os folds do KFold
        for fold, (train_index, test_index) in enumerate(kf.split(sequencia)):
            X_train, X_test = np.array(sequencia)[train_index], np.array(sequencia)[test_index]

            # Criar matriz de transição
            try:
                matriz_transicao, estados = calculate_transition_matrix(X_train)
                if matriz_transicao.size == 0:  # Pular se não há estados válidos
                    precisao_scores.append(0)
                    continue

                mc = MarkovChain(matriz_transicao, states=estados)
            except ValueError as e:
                print(f"Erro ao criar a matriz de transição para o veículo {veiculo}: {e}")
                precisao_scores.append(0)
                continue

            # Predições e cálculo de precision_score
            predicoes = []
            reais = X_test[1:]  # Valores reais (os próximos estados nos testes)

            for estado_inicial in X_test[:-1]:
                if estado_inicial in estados:
                    # Prevendo o próximo estado com base no estado inicial
                    proximo_estado_predito = mc.predict(initial_state=estado_inicial, steps=1)  # steps=1 para prever um estado de cada vez

                    # Garantindo que o retorno seja um valor simples (não lista ou array)
                    if proximo_estado_predito:
                        if isinstance(proximo_estado_predito, (list, np.ndarray)):
                            predicoes.append(proximo_estado_predito[0])
                        else:
                            predicoes.append(proximo_estado_predito)
                    else:
                        predicoes.append(None)
                else:
                    predicoes.append(None)


            # Filtrando valores None, pois precision_score não pode lidar com None
            predicoes_filtradas = [p for p in predicoes if p is not None]
            reais_filtrados = [r for r in reais if r is not None]

            # Verificando se as listas de predições e reais têm o mesmo tamanho
            if len(predicoes_filtradas) == len(reais_filtrados):
                # Calculando a precisão com a métrica precision_score
                precisao = precision_score(reais_filtrados, predicoes_filtradas, average='micro', zero_division=0)
            else:
                precisao = 0  # Caso o alinhamento entre predições e reais não seja correto


            precisao_scores.append(precisao)

        # Registrar resultados para o veículo
        erro_padrao = np.std(precisao_scores) / np.sqrt(len(precisao_scores))
        for fold, precisao in enumerate(precisao_scores):
            resultados1.append({"Veiculo": veiculo, "Fold": fold + 1, "Precisao": precisao, "Erro_Padrao": erro_padrao})

    # Criar DataFrame final
    df_resultados = pd.DataFrame(resultados1)
    
    # Group data by 'Veiculo' and calculate the mean of 'medias'
    df_resultados['medias'] = df_resultados.groupby('Veiculo')['Precisao'].transform('mean')
    media_por_veiculo_markov = df_resultados.groupby('Veiculo')['medias'].mean().reset_index()

# Parte do HMM:

# Função para calcular a matriz de emissão e transição
    def calcular_matriz_emissao_transicao(df):
    #     df_expandido = df.copy()  # Evita problemas de cópia
        df['observacao'] = df['turno'].astype(str) # + "_" + df_expandido['turno'].astype(str)
        df['estado_oculto'] = df['grid_origem'].astype(str)

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
    resultados2 = []

    for veiculo_id in df['VehId'].unique():  # Iterando pelos veículos únicos
        print(f"Processando veículo {veiculo_id}...")

        # Selecionar os dados do veículo
        veiculo_data = df[df['VehId'] == veiculo_id]

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        for fold, (train_index, test_index) in enumerate(kf.split(veiculo_data)):
            train_data, test_data = veiculo_data.iloc[train_index].copy(), veiculo_data.iloc[test_index].copy()

            # Criando colunas para treino
            train_data['observacao'] =  train_data['turno'].astype(str) # + "_" + train_data['turno'].astype(str)
            train_data['estado_oculto'] = train_data['grid_origem'].astype(str)

            # Criando colunas para teste
            test_data['observacao'] = test_data['turno'].astype(str) # + "_" + test_data['turno'].astype(str)
            test_data['estado_oculto'] = test_data['grid_origem'].astype(str)

            # Calculando as matrizes de emissão e transição para o conjunto de treino
            matriz_emissao_treino, matriz_transicao_treino, estados_treino, observacoes_treino = calcular_matriz_emissao_transicao(train_data)

            # Lidar com casos de apenas 1 estado
            if matriz_transicao_treino.shape[0] == 1:
                print(f"Apenas 1 estado no conjunto de treino ({estados_treino[0]}). Precisao definida como 0.0%.")
                resultados2.append({"Veiculo": veiculo_id, "Fold": fold + 1, "Precisao": 0.0, "Erro_Padrao": 0.0})
                continue

            # Verificar validade das dimensões
            if matriz_transicao_treino.shape[0] != matriz_transicao_treino.shape[1]:
                print(f"Conjunto de treino inválido: matriz de transição não é quadrada. Ignorando este fold.")
                resultados2.append({"Veiculo": veiculo_id, "Fold": fold + 1, "Precisao": 0.0, "Erro_Padrao": 0.0})
                continue

            if matriz_emissao_treino.shape[0] != len(estados_treino) or matriz_emissao_treino.shape[1] != len(observacoes_treino):
                print("Dimensões da matriz de emissão não correspondem aos estados ou observações. Ignorando este fold.")
                resultados2.append({"Veiculo": veiculo_id, "Fold": fold + 1, "Precisao": 0.0, "Erro_Padrao": 0.0})
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
                precisao = 0.0
            else:
                # Realizar predição
                y_pred = hmm.predict(prediction_type='viterbi', symbols=X_test_filtrado, initial_status=y, output_indices=False)

                # Verificar se a predição é válida
                if y_pred is None or y_pred[1] is None:
                    precisao = 0.0
                else:
                    # Filtrar estados ocultos correspondentes às observações filtradas
                    indices_validos = [i for i, symbol in enumerate(X_test) if symbol in observacoes_treino]
                    y_test_filtrado = [y_test[i] for i in indices_validos]

                    # Calcular precisão usando precision_score com média macro
                    precisao = precision_score(y_test_filtrado, y_pred[1], average='micro', zero_division=0)

            erro_padrao = np.std([precisao]) / np.sqrt(1)  # Como há apenas uma precisão por fold, o erro padrão é zero
            resultados2.append({"Veiculo": veiculo_id, "Fold": fold + 1, "Precisao": precisao, "Erro_Padrao": erro_padrao})

            erro_padrao = 0.0  # como só temos uma amostra por fold, o erro padrão será 0
            resultados2.append({
                "Veiculo": veiculo_id,
                "Fold": fold + 1,
                "Precisao": precisao,
                "Erro_Padrao": erro_padrao
            })

# Criar DataFrame final
    hmm = pd.DataFrame(resultados2)
    
    
    hmm['medias'] = hmm.groupby('Veiculo')['Precisao'].transform('mean')
    media_por_veiculo_markovhmm = hmm.groupby('Veiculo')['medias'].mean().reset_index()
    # Inicializar lista para armazenar os valores de p e médias
    p_values = []
    medias_resultados = []
    medias_markov = []

    # Calcular o valor de p para cada veículo (comparação entre os dois conjuntos de dados)
    veh_ids = df_resultados["Veiculo"].unique()
    for i in range(30):  # Para cada veículo
        # Obter as precisões do veículo i
        precisao_resultado = df_resultados[df_resultados['Veiculo'] == veh_ids[i]]['Precisao'].values
        precisao_hmm = hmm[hmm['Veiculo'] == veh_ids[i]]['Precisao'].values

        # Calcular a média para cada veículo
        medias_resultados.append(np.mean(precisao_resultado))
        medias_markov.append(np.mean(precisao_hmm))

        # Comparar as precisões do veículo i entre os dois conjuntos de dados
        _, p_value = ttest_ind(precisao_resultado, precisao_hmm)
        p_values.append(p_value)

    # Obter os 30 menores valores de p e os veículos correspondentes
    indices_menores_p = np.argsort(p_values)[:30]  # Índices dos 30 menores p-valores

    indices = np.arange(1, 31)  # 30 veículos
    width = 0.4  # Largura das barras

    # Posicionar os veículos no eixo X
    x_ticks = [str(veh_ids[i]) for i in range(30)]
    x_markov = indices - width / 2
    x_hmm = indices + width / 2

    # Criar o primeiro gráfico: Comparação de precisões por veículo (usando plotly)
    fig1 = go.Figure()

    # Trace único para Cadeias de Markov
    fig1.add_trace(go.Bar(
        x=x_markov,
        y=medias_resultados,
        name="Cadeias de Markov",
        marker=dict(color='skyblue')
    ))

    # Trace único para Cadeias Ocultas de Markov
    fig1.add_trace(go.Bar(
        x=x_hmm,
        y=medias_markov,
        name="Cadeias Ocultas de Markov",
        marker=dict(color='orange')
    ))

    # Configuração do primeiro gráfico
    fig1.update_layout(
        title="Comparação de Precisões por Veículo",
        xaxis=dict(
            title="VehId",
            tickvals=indices,
            ticktext=x_ticks,
            tickangle=45
        ),
        yaxis=dict(title="Precisão Média (%)"),
        barmode='group',
        showlegend=True
    )

    # Exibir o primeiro gráfico no Streamlit
    st.plotly_chart(fig1)

    # ------------------------------------------
    # Segundo gráfico: Comparação de p-values (ordenado por veh_ids)
    fig2 = go.Figure()

    # Ordenar p-values pelos 30 menores
    sorted_p_values = [p_values[i] for i in indices_menores_p]
    sorted_veh_ids = [veh_ids[i] for i in indices_menores_p]

    # Adicionar barras para p-values
    fig2.add_trace(go.Bar(
        x=sorted_veh_ids,
        y=sorted_p_values,
        marker=dict(color='lightcoral')
    ))

    # Configuração do segundo gráfico
    fig2.update_layout(
        title="P-Values por VehId (Ordenados)",
        xaxis=dict(
            title="VehId",
            tickvals=sorted_veh_ids,
            tickangle=45
        ),
        yaxis=dict(title="P-Value"),
        showlegend=False
    )

    # Exibir o segundo gráfico no Streamlit
    st.plotly_chart(fig2)

    # ------------------------------------------
    # Terceiro gráfico: Comparação de médias globais
    media_global_markov = df_resultados['Precisao'].mean()
    media_global_hmm = hmm['Precisao'].mean()

    fig3 = go.Figure()

    # Adicionar barras para as médias globais
    fig3.add_trace(go.Bar(
        x=["Markov", "Markov Oculto"],
        y=[media_global_markov, media_global_hmm],
        marker=dict(color=["skyblue", "orange"]),
        width=0.6
    ))

    # Configuração do terceiro gráfico
    fig3.update_layout(
        title="Comparação de Precisão Média Global",
        xaxis=dict(title="Método"),
        yaxis=dict(title="Precisão Média (%)"),
        showlegend=False
    )

    # Exibir o terceiro gráfico no Streamlit
    st.plotly_chart(fig3)
