from numpy.lib.arraysetops import unique
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
import re

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL

from dash_extensions import Lottie       # pip install dash-extensions
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
import dash_table

# ---------- Import and clean data (importing csv into pandas)
df = pd.DataFrame()

for path, currentDirectory, files in os.walk("/mnt/c/Users/Fernando/Documents/Project_2/output/mtb/prediction/results"):
    for file in files:
        if file.endswith("_cv.csv"):
            file_noext = file.replace("_cv.csv", "")
            file_split = re.split(r"-", file_noext)
            df_tmp = pd.read_csv(os.path.join(path, file))
            df_tmp.insert(0, "Model", file_split[0])
            df_tmp.insert(1, "Drug", file_split[1])
            df = pd.concat([df, df_tmp])

df['param_C_log'] = np.log10(df['param_C'])
df['param_max_iter_log'] = np.log10(df['param_max_iter'])
dfg = pd.DataFrame({'param_solver':df['param_solver'].unique()})
dfg['param_solver_ind'] = dfg.index
df = pd.merge(df, dfg, on = 'param_solver', how='left')

# df without cv-splits data
df_sub = df[df.columns.drop(df.filter(regex=("^split|rank|std")).columns)]

# Bootstrap themes by Ann: https://hellodash.pythonanywhere.com/theme_explorer
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX,'https://codepen.io/chriddyp/pen/bWLwgP.css'],
    title = 'BenchmarkDR Analytics')

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H1("Performance summary", style={'text-align': 'center'}),
                    dcc.Graph(id='performance-summary', figure={})
                ])
            ]),
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H1("Resistance status", style={'text-align': 'center'}),
                    dcc.Graph(id='res-status', figure={})
                ])
            ]),
        ], width=4),
    ],className='mb-2 mt-2'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H1("Select", style={'text-align': 'center'}),
                    dcc.Dropdown(id="my-model",
                        options=[
                            {"label": i, "value": i} for i in df["Model"].unique()],
                        multi=False,
                        value=df["Model"].unique()[0],
                        style={'width': "100%"}
                        ),
                    dcc.RadioItems(id="my-drug",
                        options=[
                            {"label": i, "value": i} for i in df["Drug"].unique()],
                        value=df["Drug"].unique()[0],
                        labelStyle={'dispay':'block'}
                        )
                ])
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H1("Hyperparameter visualization", style={'text-align': 'center'}), 
                    dcc.Graph(id='hyperparameter-viz', figure={})
                ])
            ]),
        ], width=5),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div(id='param_table', children=[])
                    # dash_table.DataTable(
                    #     id='param_table',
                    #     columns = [{"name": i, "id": i} for i in df.columns],
                    #     editable=False,
                    #     filter_action="native",
                    #     sort_action="native",
                    #     sort_mode="multi",
                    #     row_selectable="multi",
                    #     row_deletable=False,
                    #     selected_rows=[],
                    #     page_action="native",
                    #     page_current= 0,
                    #     page_size= 6
                        # page_action='none',
                        # style_cell={
                        # 'whiteSpace': 'normal'
                        # },
                        # fixed_rows={ 'headers': True, 'data': 0 },
                        # virtualization=False,
                    # ),
                ]),
            ]),
        ], width=5),
    ],className='mb-2'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H1("Cross validation results", style={'text-align': 'center'}),
                    dcc.Graph(id='cv-res', figure={})
                ])
            ]),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H1("Train-test results", style={'text-align': 'center'}),
                    dcc.Graph(id='train-test', figure={})
                ])
            ]),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H1("Features", style={'text-align': 'center'}),
                    dcc.Graph(id='features', figure={})
                ])
            ]),
        ], width=4),
    ],className='mb-2'),
], fluid=True)


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components

# @app.callback(
#     [Output(component_id='output_container', component_property='children'),
#      Output(component_id='my_parcoord', component_property='figure')],
#     [Input(component_id='slct_drug', component_property='value')]
# )

# ------------------------------------------------------------------------------
# Hyperparameter visualization
@app.callback(
    Output('hyperparameter-viz', 'figure'),
    Input('my-model', 'value'),
    Input('my-drug', 'value')
)

def update_hyperparameter_viz(model, drug):
    dff = df.copy()
    dff = dff[dff["Model"] == model]
    dff = dff[dff["Drug"] == drug]

    fig = go.Figure(data=
        go.Parcoords(
            line_color = 'blue',
            dimensions = list([
                dict(tickvals = dff['param_solver_ind'], ticktext = ['liblinear', 'saga'],
                    label = 'Solver', values = dff['param_solver_ind']),
                dict(range = [min(dff['param_C_log']), max(dff['param_C_log'])],
                    label = 'log10(C)', values = dff['param_C_log']),
                dict(range = [min(dff['param_max_iter_log']), max(dff['param_max_iter_log'])],
                    label = 'log10(Max iter)', values = dff['param_max_iter_log']),
                dict(range = [0,1],
                    label = 'Balanced accuracy', values = dff['mean_test_balanced_accuracy']),
                dict(label = 'Time', values = dff['mean_fit_time'])
            ])
        )
    )

    return fig

# ------------------------------------------------------------------------------
# Dash table

@app.callback(
    Output('param_table', 'children'),
    Input('my-model', 'value'),
    Input('my-drug', 'value'))

def update_param_table_data(model, drug):
    if model is None or model == []:
        model == MODEL
    if drug is None or drug == []:
        drug is DRUG
    param_table = df.copy()
    param_table = param_table[param_table["Model"] == model]
    param_table = param_table[param_table["Drug"] == drug]

    param_table.replace("", float("NaN"), inplace=True)
    param_table.dropna(how='all', axis=1, inplace=True)

    return dash_table.DataTable(
        id='table',
        columns = [{"name": i, "id": i} for i in param_table.filter(regex=("^param")).columns],
        data = param_table.to_dict('records'),
        style_cell={
            'textAlign': 'center',
            'color': 'black',
            'fontFamily': 'sans-serif',
        },
        editable=False,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_selectable="multi",
        row_deletable=False,
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 6,
    )


# @app.callback(
#     Output('param_table', 'columns'),
#     Input('my-model', 'value'),
#     Input('my-drug', 'value'))

# def update_param_table_cols(model, drug):
#     if model is None or model == []:
#         model == MODEL
#     if drug is None or drug == []:
#         drug is DRUG
#     param_table = df.copy()
#     param_table = param_table[param_table["Model"] == model]
#     param_table = param_table[param_table["Drug"] == drug]

#     cols = [{"name": i, "id": i} for i in param_table.columns]

#     return cols
    


# ------------------------------------------------------------------------------
# Line plot

@app.callback(
    Output('dropdown-container', 'children'),
    Input('add-filter', 'n_clicks'),
    State('dropdown-container', 'children'))
def display_dropdowns(n_clicks, children):
    new_dropdown = dcc.Dropdown(
        id={
            'type': 'filter-dropdown',
            'index': n_clicks
        },
        options=[{'label': i, 'value': i} for i in ['NYC', 'MTL', 'LA', 'TOKYO']]
    )
    children.append(new_dropdown)
    return children


@app.callback(
    Output('line-plot', 'figure'),
    Input('my-model', 'value'),
    Input('my-drug', 'value'),
    Input('y-axis', 'value'),
    Input('x-axis', 'value'),
    Input('group', 'value'),
    Input('filter', 'value')
)

def update_line_plot(model, drug, y, x, group, filter):
    dff = df.copy()
    dff = dff[dff["Model"] == model]
    dff = dff[dff["Drug"] == drug]
    
    dff = dff[[y, x, group]]
    dff_group = dff.groupby(dff[group])

    labels = dff[group].unique()

    fig = go.Figure()

    for i in range (0,len(labels)):
        plot_data = dff_group.get_group(labels[i])
        fig.add_trace(go.Scatter(x=plot_data[x], y=plot_data[y], mode='lines+markers',
            name=labels[i]
        ))

    return fig

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
