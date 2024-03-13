#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:44:49 2024

@author: joelgagnon
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import pickle



# Assuming you have your database variable defined with your data
#%% Get the sample data imported

# File path where the pickled database is saved
file_path = "test_data/CanadianChamps2023/database2"
file_path = "test_data/EliteCanada2024/jrdatabase"
# file_path = "test_data/EliteCanada2024/srdatabase"
# Load the pickled database
with open(file_path, 'rb') as f:
    database = pickle.load(f)

print("Database loaded successfully.")
#%% 



# Create Dash app
app = dash.Dash(__name__)

# Define layout of the app
app.layout = html.Div([
    dcc.Dropdown(
        id='athlete-dropdown',
        options=[{'label': name, 'value': name} for name in database.keys()],
        value=list(database.keys())[0]
    ),
    dcc.Dropdown(
        id='result-dropdown',
        multi=True,
    ),
    dcc.Dropdown(
        id='y-axis-dropdown',
        options=[
            {'label': 'D', 'value': 'D'},
            {'label': 'Score', 'value': 'Score'},
            {'label': 'Rk', 'value': 'Rk'},
            {'label': 'E', 'value': 'E'}
        ],
        value='D'
    ),
    dcc.Graph(id='individual-graph')
])

# Define callback to update result dropdown based on selected athlete
@app.callback(
    Output('result-dropdown', 'options'),
    [Input('athlete-dropdown', 'value')]
)
def update_result_dropdown(selected_athlete):
    return [{'label': day, 'value': day} for day in database[selected_athlete].keys()]

# Define callback to update graph based on selected options
@app.callback(
    Output('individual-graph', 'figure'),
    [Input('athlete-dropdown', 'value'),
     Input('result-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')]
)
def update_graph(selected_athlete, selected_results, y_axis):
    if selected_athlete is None or selected_results is None:
        # If no athlete or result selected, return an empty graph
        return {'data': [], 'layout': {'title': 'Select athlete and results to display'}}

    traces = []
    for result in selected_results:
        x_values = [key for key in database[selected_athlete][result].keys() if key != "AA"]
        y_values = [database[selected_athlete][result][x][y_axis] for x in x_values]
        # Filter out zeros
        x_values = [x if x != 0.0 else np.nan for x in x_values]
        y_values = [y if y != 0.0 else np.nan for y in y_values]

        trace = go.Scatter(x=x_values, y=y_values, mode='lines+markers', name=result)
        traces.append(trace)

    layout = go.Layout(
        title=f'{selected_athlete} - {", ".join(selected_results)} ({y_axis})',
        xaxis=dict(title='Apparatus'),
        yaxis=dict(title=y_axis),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return {'data': traces, 'layout': layout}


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
