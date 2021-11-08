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
from dash.dependencies import Input, Output


app = dash.Dash(__name__)


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

df = df[df["Model"] == "SVM_l2"]

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("Web Application Dashboards with Dash", style={'text-align': 'center'}),

    dcc.Dropdown(id="slct_drug",
                 options=[
                     {"label": i, "value": i}
                     for i in df["Drug"].unique()],
                 multi=False,
                 placeholder="Select a drug",
                 style={'width': "40%"}
                 ),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='my_parcoord', figure={})

])

# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components

# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='my_parcoord', component_property='figure')],
    [Input(component_id='slct_drug', component_property='value')]
)

def update_graph(option_slctd):
    print(option_slctd)

    container = "The drug you selected was: {}".format(option_slctd)
    
    dff = df.copy()
    dff = dff[dff["Drug"] == option_slctd]

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

    return container, fig


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
