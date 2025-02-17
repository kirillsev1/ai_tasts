import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import dash_table

import requests
import pandas as pd
from io import StringIO
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

"""READ DATA"""

response = requests.get('http://asterank.com/api/kepler?query={}&limit=2000')

df = pd.json_normalize(response.json())
df = df[df['PER'] > 0]
print(df)
df['KOI'] = df['KOI'].astype(int, errors='ignore')

#create Star size category
bins = [0, 0.8, 1.2, 100]
names = ['small', 'similar', 'bigger']
df['StarSize'] = pd.cut(df['RSTAR'], bins, labels=names)  #разбили колоноку RSTAR на бины с заданными названиями

#TEMPERATURE BINS
tp_bins = [0, 200, 400, 500, 5000]
tp_labels = ['low', 'optimal', 'high', 'extreme']
df['temp'] = pd.cut(df['TPLANET'], tp_bins, labels=tp_labels)  #переменная которая говорит приемлима ли планета по t

#SIZE BINS
rp_bins = [0, 0.5, 2, 4, 100]
rp_labels = ['low', 'optimal', 'high', 'extreme']
df['gravity'] = pd.cut(df['RPLANET'], rp_bins, labels=rp_labels)  # приемлима ли планета по размеру

#ESTIMATE OBJECT STATUS
df['status'] = np.where((df['temp'] == 'optimal') &
                        (df['gravity'] == 'optimal'),
                        'promising', None)
df.loc[:,'status'] = np.where((df['temp'] == 'optimal') &
                                (df['gravity'].isin(['low', 'high'])),
                                'challenging', df['status'])
df.loc[:,'status'] = np.where((df['gravity'] == 'optimal') &
                                (df['temp'].isin(['low', 'high'])),
                                'challenging', df['status'])
df['status'] = df.status.fillna('exrteme')

#RELATIVE DISTANCE (distance to SUN/SUM radii)
df.loc[:,'relative_dist'] = df['A']/df['RSTAR']   #показывает насколько др.планеты находится близко\далеко от своего солнца

#GLOBAL DESIGN SETTINGS (шаблоны для стиля и цвета графиков)
CHARTS_TEMPLATE = go.layout.Template(   #задаем объект в начале приложения и внутри задаем параметры (шаблон для стиля графиков)
    layout=dict(      #внутри словаря задаем настройки
        font=dict(family='Century Gothic', #настройки шрифта по центру
                size=14),
        legend=dict(orientation='h',   #горизонтально (легенда)
                    title_text='',     #без заголовка
                    x=0,                #задаем положение через координаты
                    y=1.1)
    )
)

COLOR_STATUS_VALUES = ['lightgray', '#1F85DE', '#f90f04']  #шаблон для изменения цветов легенд

#FILTERS
options = []
for k in names:  #по names проходимся и берем значения для label и value
    options.append({'label': k, 'value': k})

star_size_selector = dcc.Dropdown(
    id='star-selector', #айди для обращенния
    options=options, #значения которыем можно выбирать
    value=['small', 'similar', 'bigger'], #дефолтные значения(изначальные)
    multi=True  #параметр что можно выбирать несколько значений
)

rplanet_selector = dcc.RangeSlider(
    id='range-slider',
    min=min(df['RPLANET']),
    max=max(df['RPLANET']),
    marks={5: '5', 10: '10', 20: '20'},   #создаем метки в виде словаря ключ:значение
    step=1,
    value=[min(df['RPLANET']), max(df['RPLANET'])] #значения стлайдера по дефолту
)
## НОВОЕ # идентификатор temp-mstar-chart тут и во втором @app.callback()
#TABS CONTENT (создаем вкладку с графиками)
tab1_content = [dbc.Row([    # строка графиков
        dbc.Col(html.Div(id='dist-temp-chart'), md=6),
        dbc.Col(html.Div(id='celestial-chart'), md=6)
    ], style={'margin-top': 20}),  #отступ от табов сверху
                dbc.Row([
        dbc.Col(html.Div(id='relative-dist-chart'), md=6),
        dbc.Col(html.Div(id='mstar-tstar-chart'), md=6),
        dbc.Col(html.Div(id='temp-mstar-chart'), md=6), # здесь тоже добавили уникальный идентификатор графика зависимости масса звезды ~ температура планеты
    ])]

tab2_content = [dbc.Row(html.Div(id='data-table'), style={'margin-top': 20})]

#tab3 content
table_header = [ #заголовок таблицы
    html.Thead(html.Tr([html.Th("Field name"), html.Th("Details")]))
]

# Параметры
expl = {'KOI': 'Object of Interest number',  #создали словарь для таблички будущей
        'A': 'Semi-major axis (AU)',
        'RPLANET': 'Planetary radius (Earth radii)',
        'RSTAR': 'Stellar radius (Sol radii)',
        'TSTAR': 'Effective temperature of host star as reported in KIC (k)',
        'KMAG': 'Kepler magnitude (kmag)',
        'TPLANET': 'Equilibrium temperature of planet, per Borucki et al. (k)',
        'T0': 'Time of transit center (BJD-2454900)',
        'UT0': 'Uncertainty in time of transit center (+-jd)',
        'PER': 'Period (days)',
        'UPER': 'Uncertainty in period (+-days)',
        'DEC': 'Declination (@J200)',
        'RA': 'Right ascension (@J200)',
        'MSTAR': 'Derived stellar mass (msol)'}

tbl_rows = []
for i in expl:     #перебираем ключи словаря созданного для заполнения строк таблицы
    tbl_rows.append(html.Tr([html.Td(i), html.Td(expl[i])]))
table_body = [html.Tbody(tbl_rows)]
table = dbc.Table(table_header + table_body, bordered=True)


text = 'Data are sourced from Kepler API via asterank.com'
tab3_content = [dbc.Row(html.A(text, href='https://www.asterank.com/kepler'),
                        style={'margin-top': 20}),
                dbc.Row(html.Div(children=table),
                        style={'margin-top': 20})]


app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.FLATLY])  #показываем что мы используем внешние стили(тему-белый фон) для приложения

"""LAYOUT"""  # отписывает то, как выглядит приложение
app.layout = html.Div([   # строка заголовков
    #HEADER
    dbc.Row([
        dbc.Col(
            html.Img(src=app.get_asset_url('images/planets.png'), #пишим путь картинки и вставляем ее
                    style={'width': '100px', 'marginLeft': '40px'}),  # указываем ширину картинки
            width={'size': 1} #размер одна калонка
        ),
        dbc.Col([
            html.H1("Exoplanet Data Visualization"),
            html.A('Read about exoplanets', href='https://spaceplace.nasa.gov/')],
            width={'size': 7}),
        dbc.Col(html.Div([
            html.P('Developed by '),
            html.A('Yana Novikova', href='', style={'marginLeft': '3px'})
        ], className='app-referral'),
                width={'size': 4})],
                className='app-header'),
    dcc.Store(id='filtered-data', storage_type='session'),
    #BODY
    html.Div([
        #FILTERS
        dbc.Row([               # строка фильтров
            dbc.Col([
                html.H6('Select planet main semi-axis range'), #название фильтра
                html.Div(rplanet_selector) #сам фильтр
            ],
                width={'size': 2}),
            dbc.Col([
                html.H6('Star size'),  # название фильтра
                html.Div(star_size_selector)
            ],
                width={'size': 3, 'offset': 1}),  #ширина и отступ
            dbc.Col(dbc.Button('Apply', id='submit-val', n_clicks=0,  #добавим кнопку применить
                            className='mr-2', style={'margin-top': '20px'}))  #стиль кнопки
        ]),
        #CHARTS
        dbc.Tabs([
            dbc.Tab(tab1_content, label='Charts'),
            dbc.Tab(tab2_content, label='Data'),
            dbc.Tab(tab3_content, label='About')
        ])
        ],
        className='app-body') #отступы от краев страницы задаем через класс теперь CSS
])


"""CALLBACKS""" #описывает интеррактивность приложения

@app.callback(
    Output(component_id='filtered-data', component_property='data'),
        [Input(component_id='submit-val', component_property='n_clicks')],  #для кнопки
        [State(component_id='range-slider', component_property='value'), #для значений фильтров которые передаем в график
        State(component_id='star-selector', component_property='value')]
)
def filter_data(n, radius_range, star_size):
    print(n, radius_range, star_size)
    my_data = df[(df['RPLANET'] > radius_range[0]) &
                    (df['RPLANET'] < radius_range[1]) &
                    (df['StarSize'].isin(star_size))]
    return my_data.to_json(date_format='iso', orient='split',
                        default_handler=str)


@app.callback(         #для динамического изменения нашего графика
    [Output(component_id='dist-temp-chart', component_property='children'),
            Output(component_id='celestial-chart', component_property='children'),
            Output(component_id='relative-dist-chart', component_property='children'),
            Output(component_id='mstar-tstar-chart', component_property='children'),
            Output(component_id='data-table', component_property='children')],
            Output(component_id='temp-mstar-chart', component_property='children'), #добавляем здесь Output с меткой (название зависит от параметров)
            [Input(component_id='filtered-data', component_property='data')]

)
def update_dist_temp_chart(data):
    chart_data = pd.read_json(StringIO(data), orient='split')
    if len(chart_data) == 0:
        return html.Div('Please select more data'), html.Div(), html.Div(), html.Div(), html.Div(), html.Div()

    fig1 = px.scatter(chart_data, x='TPLANET', y='A', color='status',
                    color_discrete_sequence=COLOR_STATUS_VALUES)  #строим график обновленный
    fig1.update_layout(template=CHARTS_TEMPLATE)
    html1 = [html.H4('Planet Temperature ~ Distance from the Star'),
            dcc.Graph(figure=fig1)]

    fig2 = px.scatter(chart_data, x='RA', y='DEC', size='RPLANET',
                    color='status', color_discrete_sequence=COLOR_STATUS_VALUES)  # строим график обновленный
    fig2.update_layout(template=CHARTS_TEMPLATE)
    html2 = [html.H4('Position on the Celestial Sphere'),
            dcc.Graph(figure=fig2)]

    #RELATIVE DIST CHART
    fig3 = px.histogram(chart_data, x='relative_dist',
                        color='status', barmode='overlay', marginal='box',
                        color_discrete_sequence=COLOR_STATUS_VALUES)
    fig3.add_vline(x=1, y0=0, y1=160, annotation_text='Earth', line_dash='dot')
    fig3.update_layout(template=CHARTS_TEMPLATE)
    html3 = [html.H4('Rlative Distance(AU/Sol radii)'),
            dcc.Graph(figure=fig3)]
    fig4 = px.scatter(chart_data, x='MSTAR', y='TSTAR', size='RPLANET',
                    color='status', color_discrete_sequence=COLOR_STATUS_VALUES)
    fig4.update_layout(template=CHARTS_TEMPLATE)
    html4 = [html.H4('Star Mass ~ Star Temperature'),
            dcc.Graph(figure=fig4)]

    fig5 = px.scatter(chart_data, x='MSTAR', y='TPLANET',
                    color='status', color_discrete_sequence=COLOR_STATUS_VALUES)
    fig5.update_layout(template=CHARTS_TEMPLATE)
    html5 = [html.H4('Star Mass ~ Planet Temperature'),
            dcc.Graph(figure=fig5)]

    #RAW DATA TABLE
    raw_data = chart_data.drop(['relative_dist', 'StarSize',
                                'ROW', 'temp', 'gravity'], axis=1) #удаляем колонки которые мы создавали сами
    tbl = dash_table.DataTable(data=raw_data.to_dict('records'), #создаем табл. конвертируя его в словарь
                            columns=[{'name': i, 'id': i}  #задали название и айди колонки (преобразуем в список словарей)
                                        for i in raw_data.columns], #берем это из названий колонок нашего датафрейма
                            style_data={'width': '100px', #задаем ширину
                                        'maxWidth': '100px',
                                        'minWidth': '100px'},
                            style_header={'textAlign': 'center'},
                            page_size=40) #ограничим количество строк
    html6 = [html.H4('Raw Data'), tbl] 

    return html1, html2, html3, html4, html6, html5


if __name__ == '__main__':  # запущен ли скрипт напрямую, а не импортирован как модуль в другой скрипт.
    app.run_server(debug=True)  # debug=True это запуск в тестовом режиме
