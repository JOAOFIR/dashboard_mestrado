import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.express as px
import folium
from folium.plugins import Search
# from streamlit_folium import st_folium
from streamlit_option_menu import option_menu

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
        options=["Mapa de Trajetórias", "Gráfico de Barras Temporais", "Tabela de Repetições Duplicadas"],
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

# Gráfico de Barras
if selected == "Gráfico de Barras Temporais":
    st.header("Gráfico de Barras de Repetições de Valores")

    # Selecionando a coluna para análise
    campo = st.selectbox("Selecione o campo para contar valores repetidos:", ['day_x', 'period_x'])

    if campo:
        # Contando quantas vezes cada valor único aparece na coluna selecionada
        contagem = grid_origem[campo].value_counts().reset_index()
        contagem.columns = ['Valor', 'Repetições']

        # Criando o gráfico de barras com as repetições no eixo X e os valores no eixo Y
        fig = px.bar(contagem, y='Valor', x='Repetições', title=f'Número de Contagem para {campo}',
                     labels={'Valor': 'Valor', 'Repetições': 'Número'},
                     text='Repetições',  # Adiciona os valores de repetição sobre as barras
                     color='Repetições',  # Aplica as cores de acordo com o valor das repetições
                     color_continuous_scale='Viridis',  # Escala de cores de frio (azul) para quente (amarelo)
                     orientation='h')  # Para deixar as barras horizontais

        # Ajustando o layout para garantir que as barras fiquem visíveis e com o texto completo
        fig.update_layout(
            xaxis_title='Número de Contagem',
            yaxis_title='Valor',
            margin=dict(l=40, r=40, t=40, b=100),  # Ajusta margens para não cortar os rótulos
            bargap=0.05,  # Ajusta o espaço entre as barras, reduzindo para 5% para deixá-las mais próximas
            height=600,  # Ajusta o tamanho do gráfico
            showlegend=False,  # Remover a legenda
            autosize=True  # Ajusta automaticamente o tamanho do gráfico
        )

        # Garantir que os rótulos do eixo Y (strings) apareçam completos
        fig.update_yaxes(tickmode='array', tickvals=contagem['Valor'])

        # Garantir que os rótulos do eixo X não apareçam com abreviações (como '20k')
        fig.update_xaxes(tickformat="d")  # Formatação de números como inteiros, sem abreviação

        # Ajustando a largura das barras, tornando-as mais espessas
        fig.update_traces(marker=dict(line=dict(width=0.5, color='black')))  # Contorno das barras para dar mais visibilidade

        # Exibindo o gráfico
        st.plotly_chart(fig)
    else:
        st.warning("Por favor, selecione um campo para análise.")

# Exibindo a tabela de repetições para o campo 'grid_origem', apenas com duplicados
if selected == "Tabela de Repetições Duplicadas":
    st.header("Tabela de Repetições Duplicadas para 'grid_origem'")

    # Contando quantas vezes cada valor único aparece na coluna 'grid_origem'
    contagem = grid_origem['grid_origem'].value_counts()

    # Filtrando apenas os valores com mais de uma ocorrência (duplicados)
    contagem_duplicada = contagem[contagem > 1].reset_index()
    contagem_duplicada.columns = ['Valor', 'Repetições']
    
    # Exibindo a tabela com os valores duplicados
    st.dataframe(contagem_duplicada)  # Exibe a tabela com as repetições duplicadas
