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
from dash.exceptions import PreventUpdate

from dash_extensions import Lottie       # pip install dash-extensions
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
import dash_table

# ---------- Import summary.csv for all performance summary (importing csv into pandas)
summary = pd.read_csv("/mnt/c/Users/Fernando/Documents/Project_2/output/mtb/prediction/summary.csv")

cols = summary.columns
unwanted_metric = ['Model', 'Drug', 'tp', 'tn', 'fp', 'fn'] # for now this is the working solution to not display certain metric
metrics = [col for col in cols if col not in unwanted_metric]


# ---------- Import labels (importing csv into pandas)
# labels = pd.DataFrame()
# for path, currentDirectory, files in os.walk("/mnt/c/Users/Fernando/Documents/Project_2/output/mtb/prediction/labels"):
#     for file in files:
#         if file.startswith("label_"):
#             drug = file.replace("label_", "").replace(".csv", "")
#             df_tmp = pd.read_csv(os.path.join(path, file))
#             labels = pd.concat([labels, df_tmp])

labels = pd.read_csv("/mnt/c/Users/Fernando/Documents/Project_2/output/mtb/prediction/labels/labels.csv")
resistant = labels.sum(axis = 0, skipna = True)
total = labels.count(axis = 0)

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
                    dcc.Graph(id='performance-summary', figure={}),
                    dcc.RadioItems(id="slct-metric",
                        options=[{"label": i, "value": i} for i in metrics],
                        value = metrics[1],
                        labelStyle={'dispay': 'inline-block'}
                    )
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
                    html.H1("Best performance table", style={'text-align': 'center'}),
                    html.Div(id="summary-table"),
                ])
            ]),
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H1("Resistance correlation", style={'text-align': 'center'}),
                    dcc.Graph(id='res-corr', figure={})
                ])
            ]),
        ], width=4),
    ],className='mb-2 mt-2'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H1("Select", style={'text-align': 'center'}),
                    html.P("Select a model:"),
                    dcc.Dropdown(id="my-model",
                        options=[
                            {"label": i, "value": i} for i in df["Model"].unique()],
                        multi=False,
                        value=df["Model"].unique()[0],
                        style={'width': "100%"}
                        ),
                    html.P("Select a drug:"),
                    dcc.RadioItems(id="my-drug",
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
                    html.H1("Table of parameters", style={'text-align': 'center'}),
                    html.Div(id="param-table"),
                ]),
            ]),
        ], width=5),
    ],className='mb-2'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H1("Select", style={'text-align': 'center'}),

                    html.P("Select train/test:"),
                    dcc.RadioItems(id='slct-train-test',
                    options=[{"label": i, "value": i} for i in ["test", "train"]], value="test",
                    labelStyle={'display': 'block'}),
                    html.Br(),
                    
                    html.P("Select y-axis:"),
                    dcc.RadioItems(id='slct-y-axis',labelStyle={'display': 'block'}),
                    html.Br(),

                    html.P("Select x-axis:"),
                    dcc.RadioItems(id="slct-x-axis", labelStyle={'display': 'block'}),
                    html.Br(),

                    html.P("Select grouping parameter:"),
                    dcc.RadioItems(id="slct-group")
                ])
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H1("Diagnostic plot", style={'text-align': 'center'}),
                    dcc.Graph(id='plot-diagnostics', figure={})
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H1("Cross validation results", style={'text-align': 'center'}),
                    dcc.RadioItems(id="slct-cv-option"),
                    dcc.Graph(id='cv-res', figure={})
                ])
            ]),
        ], width=6)
    ]),
    # dbc.Row([
    #     dbc.Col([
    #         dbc.Card([
    #             dbc.CardBody([
    #                 html.H1("Train-test results", style={'text-align': 'center'}),
    #                 dcc.Graph(id='train-test', figure={})
    #             ])
    #         ]),
    #     ], width=4),
    #     dbc.Col([
    #         dbc.Card([
    #             dbc.CardBody([
    #                 html.H1("Features", style={'text-align': 'center'}),
    #                 dcc.Graph(id='features', figure={})
    #             ])
    #         ]),
    #     ], width=4),
    # ],className='mb-2'),
    dcc.Store(id='intermediate-data'),
], fluid=True)


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components

# ------------------------------------------------------------------------------
# Plotting performance summary scatterplot based on selected metric

@app.callback(
    Output('performance-summary', 'figure'),
    Input('slct-metric', 'value')
)
def display_performance_summary(metric):
    dfc = summary.copy()
    fig = px.scatter(dfc, x=metric, y="Drug", color="Model")

    return fig


# ------------------------------------------------------------------------------
# Performance summary table

@app.callback(
    Output('summary-table', 'children'),
    Input('slct-metric', 'value')
)

def update_param_table_data(metric):
    dfc = summary.copy().round(3)
    idx = dfc.groupby("Drug")[metric].transform(max) == dfc[metric]
    dfc = dfc[idx].sort_values("Drug")

    cols = ["Drug", "Model"]
    cols.extend(metrics)
    columns = [{"name": i, "id": i} for i in cols]

    return dash_table.DataTable(
            data=dfc.to_dict('records'),
            columns = columns,
            style_cell={
                'textAlign': 'center',
                'color': 'black',
                'fontFamily': 'sans-serif',
            },
            editable=False,
            sort_action="native",
            sort_mode="multi",
            row_deletable=False,
            page_action="native",
            page_current= 0,
            page_size= 7,
        )

# ------------------------------------------------------------------------------
# Display dropdown of drugs based on selected model

@app.callback(
    [Output('my-drug', 'options'),
    Output('my-drug', 'value')],
    Input('my-model', 'value')
)
def display_drugs(model):
    dfc = df.copy()
    dfc = dfc[dfc["Model"] == model]

    options=[{"label": i, "value": i} for i in dfc["Drug"].unique()]
    value=dfc["Drug"].unique()[0]

    return options, value

# ------------------------------------------------------------------------------
# Storing temporary dataset

@app.callback(
    Output('intermediate-data', 'data'),
    Input('my-model', 'value'),
    Input('my-drug', 'value')
)
def select_data(model, drug):
    dfc = df.copy()

    if model is None or model == []:
        model == MODEL
    if drug is None or drug == []:
        drug is DRUG

    dfc = dfc[dfc["Model"] == model]
    dfc = dfc[dfc["Drug"] == drug]
    dfc.replace("", float("NaN"), inplace=True)
    dfc.dropna(how='all', axis=1, inplace=True)
    dfc = dfc.loc[:, (dfc != 0).any(axis=0)]

    return dfc.to_dict('records')

# ------------------------------------------------------------------------------
# Hyperparameter visualization
@app.callback(
    Output('hyperparameter-viz', 'figure'),
    Input('intermediate-data', 'data')
)

def update_hyperparameter_viz(data):
    if data is None:
        raise PreventUpdate
    dfc = pd.DataFrame.from_dict(data)

    dlist = []
    for col in dfc.filter(regex=("^param_")).columns:
        dic = dict(label = col, values = dfc[col])
        dlist.append(dic)
    dlist.append(dict(range = [0,1], label = 'Balanced accuracy', values = dfc['mean_test_balanced_accuracy']))
    dlist.append(dict(label = 'Time', values = dfc['mean_fit_time']))

    fig = go.Figure(data=
        go.Parcoords(
            line_color = 'blue',
            dimensions = dlist
        )
    )

    return fig

# ------------------------------------------------------------------------------
# Param table

@app.callback(
    Output('param-table', 'children'),
    Input('intermediate-data', 'data')
)

def update_param_table_data(data):
    if data is None:
        raise PreventUpdate

    dfc = pd.DataFrame.from_dict(data)
    
    columns = [{"name": i, "id": i} for i in dfc.filter(regex=("Drug|^param_")).columns]

    return dash_table.DataTable(
            data=data,
            columns=columns,
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

# ------------------------------------------------------------------------------
# Sets of diagnostics plots (accuracy, balanced-accuracy, f1score, roc-auc)

@app.callback(
    [Output('slct-y-axis', 'options'),
    Output('slct-y-axis', 'value')],
    Input('intermediate-data', 'data')
)
def display_cv_res_checklist(data):
    dfc = pd.DataFrame.from_dict(data)
    cols = dfc.filter(regex=("^split\d+_test_")).columns
    cols = [re.sub("^split\d+_test_", "", col) for col in cols]
    cols = list(set(cols))
    unwanted_metric = ['tp', 'tn', 'fp', 'fn'] # for now this is the working solution to not display certain metric
    cols = [col for col in cols if col not in unwanted_metric]

    options=[{"label": i, "value": i} for i in cols]
    value=cols[0]

    return options, value

@app.callback(
    [Output('slct-x-axis', 'options'),
    Output('slct-x-axis', 'value')],
    Input('intermediate-data', 'data')
)
def display_x_axis(data):
    dfc = pd.DataFrame.from_dict(data)
    
    options=[{"label": i, "value": i} for i in dfc.filter(regex=("^param_")).columns]
    value=dfc.filter(regex=("^param_")).columns[0]

    return options, value

@app.callback(
    [Output('slct-group', 'options'),
    Output('slct-group', 'value')],
    Input('intermediate-data', 'data'),
    Input('slct-x-axis', 'value')
)
def display_group(data, x_val):
    dfc = pd.DataFrame.from_dict(data)
    dfc = dfc.drop(columns=[x_val])

    options=[{"label": i, "value": i} for i in dfc.filter(regex=("^param_")).columns]
    value=dfc.filter(regex=("^param_")).columns[0]

    return options, value

@app.callback(
    Output('plot-diagnostics', 'figure'),
    Input('intermediate-data', 'data'),
    Input('slct-train-test', 'value'),
    Input('slct-y-axis', 'value'),
    Input('slct-x-axis', 'value'),
    Input('slct-group', 'value')
)
def update_plot_diagnostics(data, train_test, y_val, x_val, group):
    dfc = pd.DataFrame.from_dict(data)
    y_lab = "mean_"+train_test+"_"+ y_val
    labels = dfc[group].unique()

    dfc = dfc[[y_lab, x_val, group]]
    dfc = dfc.groupby(dfc[group])
    
    fig = go.Figure()

    for label in labels:
        plot_data = dfc.get_group(label)
        fig.add_trace(go.Scatter(x=plot_data[x_val], y=plot_data[y_lab], mode='lines+markers',
            name=str(label)
        ))
    fig.update_layout(xaxis_title=x_val,
                   yaxis_title=y_lab)

    return fig

# ------------------------------------------------------------------------------
# Crossval box plot

@app.callback(
    Output('cv-res', 'figure'),
    Input('intermediate-data', 'data'),
    Input('slct-y-axis', 'value')
)
def update_plot_cv_res(data, y_val):
    dfc = pd.DataFrame.from_dict(data)

    plot_data = dfc.filter(regex=("params|split\d+_"+"test"+"_"+y_val))
    
    fig = go.Figure()
    y = plot_data.drop(columns = 'params')
    y['mean'] = y.mean(axis=1)
    y_sorted = y.sort_values(by='mean', ascending=False)
    y_sorted = y_sorted.drop(columns='mean').values.tolist()

    for i in range(0,len(y)):
        fig.add_trace(go.Box(y = y_sorted[i], name=plot_data['params'][i],
        showlegend = False))

    fig.update_layout(xaxis_title='params',
                   yaxis_title=y_val,
                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   hoverformat='closest'))

    return fig

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
