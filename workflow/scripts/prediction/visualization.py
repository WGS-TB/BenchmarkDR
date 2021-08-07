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

# from dash_extensions import Lottie       # pip install dash-extensions
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
import dash_table

# ---------- Import summary.csv for all performance summary (importing csv into pandas)
summary = pd.read_csv("data/visualize_test_data/summary.csv")

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

labels = pd.read_csv("data/visualize_test_data/labels.csv")
labels.set_index(labels.columns[0], inplace=True, drop=True)
res_status_data = pd.DataFrame({"Resistant": labels.sum(axis = 0, skipna = True),
                                "Total": labels.count(axis = 0)})
res_status_data["Susceptible"] = res_status_data["Total"] - res_status_data["Resistant"]
res_status_data = res_status_data.sort_values("Total")

labels_corr = labels.corr()

# ------------------------------------------------------------------------------
# Summary statistics based on resitance labels data
fig_res_status = go.Figure()
fig_res_status.add_trace(go.Bar(
    y = res_status_data.index,
    x = res_status_data["Resistant"],
    name = 'Resistant',
    orientation='h'
))
fig_res_status.add_trace(go.Bar(
    y = res_status_data.index,
    x = res_status_data["Susceptible"],
    name = 'Susceptible',
    orientation='h'
))

annotations = []
for i in range(0, len(res_status_data)):
    annotations.append(dict(y=res_status_data.index[i], x=res_status_data['Total'][i]+1000,
                            text = 'Res: ' + str(round(res_status_data['Resistant'][i]/res_status_data['Total'][i]*100,1)) + '%',
                            font=dict(family='Arial', size=12),
                            showarrow=False))

fig_res_status.update_layout(barmode='stack', template="plotly_dark",
                            plot_bgcolor= 'rgba(0, 0, 0, 0)',
                            paper_bgcolor= 'rgba(0, 0, 0, 0)',
                            xaxis=dict(gridcolor="grey"),
                            annotations=annotations)

fig_labels_corr = go.Figure(data=go.Heatmap(
                                z=labels_corr,
                                x=labels_corr.index,
                                y=labels_corr.columns,
                                hoverongaps=False))
fig_labels_corr.update_layout(template="plotly_dark",
                            plot_bgcolor= 'rgba(0, 0, 0, 0)',
                            paper_bgcolor= 'rgba(0, 0, 0, 0)')

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

# df['param_C_log'] = np.log10(df['param_C'])
# df['param_max_iter_log'] = np.log10(df['param_max_iter'])
# dfg = pd.DataFrame({'param_solver':df['param_solver'].unique()})
# dfg['param_solver_ind'] = dfg.index
# df = pd.merge(df, dfg, on = 'param_solver', how='left')

# Bootstrap themes by Ann: https://hellodash.pythonanywhere.com/theme_explorer
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX, dbc.themes.DARKLY,'https://codepen.io/chriddyp/pen/bWLwgP.css'],
    title = 'BenchmarkDR Analytics')

app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("BenchmarkDR Analytics", style={'text-align': 'left', "color": "white"})
                    ])
                ], color="dark", inverse=True, style={'borderRadius':'20px'}),
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Performance summary", style={'text-align': 'left', "color": "white"}),
                        dcc.Graph(id='performance-summary', figure={}),
                        dcc.RadioItems(id="slct-metric",
                            options=[{"label": i, "value": i} for i in metrics],
                            value = metrics[1],
                            labelStyle={'dispay': 'inline-block'}
                        ),
                        html.Br(),
                        html.Div(id="summary-table"),
                    ])
                ], color="dark", inverse=True, style={'borderRadius':'20px'}),
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Resistance status", style={'text-align': 'left', "color": "white"}),
                        dcc.Graph(id='res-status', figure=fig_res_status),
                        html.Br(),
                        dcc.Graph(id='res-corr', figure=fig_labels_corr)
                    ])
                ], color="dark", inverse=True, style={'borderRadius':'20px'}),
            ], width=4),
        ],className='mb-2 mt-2'),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Select", style={'text-align': 'left', "color": "white"}),
                        html.P("Select a model:"),
                        dcc.Dropdown(id="my-model",
                            options=[
                                {"label": i, "value": i} for i in df["Model"].unique()],
                            multi=False,
                            value=df["Model"].unique()[0],
                            style={'width': "100%",
                            'backgroundColor': 'black', 'color': 'black'}
                            ),
                        html.Br(),
                        html.P("Select a drug:"),
                        dcc.RadioItems(id="my-drug",
                            labelStyle={'dispay':'block'}
                        )
                    ])
                ], color="dark", inverse=True, style={'borderRadius':'20px'}),
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Hyperparameter visualization", style={'text-align': 'left', "color": "white"}), 
                        dcc.Graph(id='hyperparameter-viz', figure={}),

                        html.P("Check metric to display:"),
                        dcc.Checklist(id="check-yaxis-hyperparameter-viz",
                                    labelStyle={'display': 'inline-block'}),
                    ])
                ], color="dark", inverse=True, style={'borderRadius':'20px'}),
            ], width=5),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Table of parameters", style={'text-align': 'left', "color": "white"}),
                        html.Div(id="param-table"),
                        
                        html.Br(),

                        html.P("Check param to log transform:"),
                        dcc.Checklist(id="param-log-transform",labelStyle={'display': 'inline-block'})
                    ]),
                ], color="dark", inverse=True, style={'borderRadius':'20px'}),
            ], width=5),
        ],className='mb-2'),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Select", style={'text-align': 'left', "color": "white"}),

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
                ], color="dark", inverse=True, style={'borderRadius':'20px'}),
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Diagnostic plot", style={'text-align': 'left', "color": "white"}),
                        dcc.Graph(id='plot-diagnostics', figure={})
                    ])
                ], color="dark", inverse=True, style={'borderRadius':'20px'})
            ], width=5),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Cross validation results", style={'text-align': 'left', "color": "white"}),
                        dcc.RadioItems(id="slct-cv-option"),
                        dcc.Graph(id='cv-res', figure={})
                    ])
                ], color="dark", inverse=True, style={'borderRadius':'20px'}),
            ], width=5)
        ]),
        # dbc.Row([
        #     dbc.Col([
        #         dbc.Card([
        #             dbc.CardBody([
        #                 html.H5("Train-test results", style={'text-align': 'left'}),
        #                 dcc.Graph(id='train-test', figure={})
        #             ])
        #         ]),
        #     ], width=4),
        #     dbc.Col([
        #         dbc.Card([
        #             dbc.CardBody([
        #                 html.H5("Features", style={'text-align': 'left'}),
        #                 dcc.Graph(id='features', figure={})
        #             ])
        #         ]),
        #     ], width=4),
        # ],className='mb-2'),
        dcc.Store(id='intermediate-data'),
        dcc.Store(id='table-data')  # TODO to relief circular dependencies of intermediate-data
    ], fluid=True)
])



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
    fig = px.scatter(dfc, x=metric, y="Drug", color="Model",
             color_discrete_sequence=px.colors.qualitative.Light24)
    fig.update_layout(template="plotly_dark",
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)')
    fig.update_traces(mode='markers', marker_line_width=2, marker_size=15)                
    fig.update_xaxes(showline=False, showgrid=False)
    fig.update_yaxes(showline=False, gridwidth=1, gridcolor='grey')

    return fig


# ------------------------------------------------------------------------------
# Performance summary table

@app.callback(
    Output('summary-table', 'children'),
    Input('slct-metric', 'value')
)

def update_summary_table_data(metric):
    dfc = summary.copy().round(3)
    idx = dfc.groupby("Drug")[metric].transform(max) == dfc[metric]
    dfc = dfc[idx].sort_values("Drug")

    cols = ["Drug", "Model"]
    cols.extend(metrics)
    columns = [{"name": i, "id": i} for i in cols]

    return dash_table.DataTable(
            data=dfc.to_dict('records'),
            columns = columns,
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'fontWeight': 'bold',
                'font_size': '15px'
            },
            style_cell={
                'backgroundColor': 'rgb(50, 50, 50)',
                'textAlign': 'left',
                'color': 'white',
                'fontFamily': 'sans-serif',
            },
            style_data_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(40, 40, 40)'
            }],
            editable=False,
            sort_action="native",
            sort_mode="multi",
            row_deletable=False,
            page_action="native",
            page_current= 0,
            page_size= 7,
            style_as_list_view=True,
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

    dfc = dfc[dfc["Model"] == model]
    dfc = dfc[dfc["Drug"] == drug]
    dfc.replace("", float("NaN"), inplace=True)
    dfc.dropna(how='all', axis=1, inplace=True)
    dfc = dfc.loc[:, (dfc != 0).any(axis=0)]

    return dfc.to_dict('records')


@app.callback(
    Output('table-data', 'data'),
    Input('intermediate-data', 'data'),
    Input('param-log-transform', 'value')
)
def select_data(data, params): # TODO update table data based on intermediate to relief 
    dfc = pd.DataFrame.from_dict(data)

    if params is not None:
        for param in params:
            dfc[param] = np.log10(dfc[param])

    return dfc.to_dict('records')


# ------------------------------------------------------------------------------
# Hyperparameter visualization
@app.callback(
    Output('hyperparameter-viz', 'figure'),
    Input('table-data', 'data'),
    Input('check-yaxis-hyperparameter-viz', 'value')
)

def update_hyperparameter_viz(data, metrics):
    if data is None:
        raise PreventUpdate
    dfc = pd.DataFrame.from_dict(data)

    dlist = []

    for col in dfc.filter(regex=("^param_")).columns:
        dic = dict(label = col, values = dfc[col])
        dlist.append(dic)
    if metrics is not None:
        for metric in metrics:
            lab = "mean_test_" + metric
            dic = dict(label = metric, values = dfc[lab])
            dlist.append(dic)

    # dlist.append(dict(range = [0,1], label = 'Balanced accuracy', values = dfc['mean_test_balanced_accuracy']))
    # dlist.append(dict(label = 'Time', values = dfc['mean_fit_time']))
    fig = go.Figure(data=
        go.Parcoords(
            labelfont = dict(size = 13),
            rangefont = dict(size = 12),
            tickfont = dict(size = 12),
            dimensions = dlist,
            line = dict(color = dfc['mean_test_roc_auc'],
                      colorscale = 'Thermal'),
            line_colorbar = dict(thickness = 12),
            legendgrouptitle_font=dict(color = 'grey')
        )
    )
    fig.update_layout(template="plotly_dark",
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)')

    return fig

@app.callback(
    [Output('check-yaxis-hyperparameter-viz', 'options'),
    Output('check-yaxis-hyperparameter-viz', 'value')],
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
    value=[cols[0]]

    return options, value

@app.callback(
    Output('param-log-transform', 'options'),
    Output('param-log-transform', 'value'),
    Input('intermediate-data', 'data')
)
def checklist_param_log_transform(data):
    dfc = pd.DataFrame.from_dict(data)
    
    options=[{"label": i, "value": i} for i in dfc.filter(regex=("^param_")).columns]
    value = [dfc.filter(regex=("^param_")).columns[0]]

    return options, value

# ------------------------------------------------------------------------------
# Param table

@app.callback(
    Output('param-table', 'children'),
    Input('table-data', 'data')
)

def update_param_table_data(data):
    if data is None:
        raise PreventUpdate

    dfc = pd.DataFrame.from_dict(data)
    
    columns = [{"name": i, "id": i} for i in dfc.filter(regex=("^param_")).columns]

    return dash_table.DataTable(
            data=data,
            columns=columns,
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'fontWeight': 'bold',
                'font_size': '15px'
            },
            style_cell={
                'backgroundColor': 'rgb(50, 50, 50)',
                'textAlign': 'left',
                'color': 'white',
                'fontFamily': 'sans-serif',
            },
            style_data_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(40, 40, 40)'
            }],
            editable=False,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            page_action="native",
            page_current= 0,
            page_size= 8,
            style_as_list_view=True,
        )

# ------------------------------------------------------------------------------
# Sets of diagnostics plots (accuracy, balanced-accuracy, f1score, roc-auc)

@app.callback(
    [Output('slct-y-axis', 'options'),
    Output('slct-y-axis', 'value')],
    Input('table-data', 'data')
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
    Input('table-data', 'data')
)
def display_x_axis(data):
    dfc = pd.DataFrame.from_dict(data)
    
    options=[{"label": i, "value": i} for i in dfc.filter(regex=("^param_")).columns]
    value=dfc.filter(regex=("^param_")).columns[0]

    return options, value

@app.callback(
    [Output('slct-group', 'options'),
    Output('slct-group', 'value')],
    Input('table-data', 'data'),
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
    Input('table-data', 'data'),
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
    fig.update_layout(template="plotly_dark",
                    xaxis_title=x_val,
                    yaxis_title=y_lab,
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    xaxis=dict(gridcolor="grey"),
                    yaxis=dict(gridcolor="grey"))

    return fig

# ------------------------------------------------------------------------------
# Crossval box plot

@app.callback(
    Output('cv-res', 'figure'),
    Input('table-data', 'data'),
    Input('slct-train-test', 'value'),
    Input('slct-y-axis', 'value')
)
def update_plot_cv_res(data, train_test, y_val):
    dfc = pd.DataFrame.from_dict(data)

    plot_data = dfc.filter(regex=("params|split\d+_"+train_test+"_"+y_val))
    
    fig = go.Figure()
    y = plot_data.drop(columns = 'params')
    y['mean'] = y.mean(axis=1)
    y_sorted = y.sort_values(by='mean', ascending=False)
    y_sorted = y_sorted.drop(columns='mean').values.tolist()

    for i in range(0,len(y)):
        fig.add_trace(go.Box(y = y_sorted[i], name=plot_data['params'][i],
        showlegend = False))

    fig.update_layout(template="plotly_dark",
                    xaxis_title='params',
                    yaxis_title=y_val,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                        hoverformat='closest'),
                    yaxis=dict(gridcolor="grey"),
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)')

    return fig

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
