#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:38:51 2024

@author: joelgagnon
"""

#%% gymcomp R1

#2024-03-26
#starting fresh -> plotly dash app

# 3 tabs with 3 key features
# 1: Competition Overview
# 2: Individual Athlete Analysis
# 3: Team Scenarios


#%% Imports

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import math
import plotly.graph_objs as go
import plotly.express as px
import pickle
import os

# Import the print function
from builtins import print
#%% Import Data 
#use absolute path

# Get the absolute path to the directory containing the main app file
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the file
file_path = os.path.join(base_dir, "test_data/EliteCanada2024/jrdatabase")


with open(file_path, 'rb') as f:
    database = pickle.load(f)

print("Database loaded successfully.")
#%%  Create Dash app
app = dash.Dash(__name__)
#%% 3 Tabs infrastructure


#%% Tab 1: Competition Overview
tab1_layout = html.Div([
    html.H3('Competition Overview'),
    dcc.Graph(
        id='graph-1',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': 'Montreal'},
            ],
            'layout': {
                'title': 'Graph 1'
            }
        }
    )
])

#%% Tab 2: Individual Athlete Analysis
tab2_layout = html.Div([
    html.H3('Individual Athlete Analysis'),
    dcc.Graph(
        id='graph-2',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [1, 4, 1], 'type': 'line', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [3, 2, 5], 'type': 'line', 'name': 'Montreal'},
            ],
            'layout': {
                'title': 'Graph 2'
            }
        }
    )
])

#%% Team Scenarios
tab3_layout = html.Div([
    html.H3('Team Scenarios'),
    dcc.Graph(
        id='graph-3',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [2, 3, 5], 'type': 'scatter', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [4, 1, 3], 'type': 'scatter', 'name': 'Montreal'},
            ],
            'layout': {
                'title': 'Graph 3'
            }
        }
    )
])

#%% Combining 3 Tabs
app.layout = html.Div([
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Tab 1', value='tab-1'),
        dcc.Tab(label='Tab 2', value='tab-2'),
        dcc.Tab(label='Tab 3', value='tab-3'),
    ]),
    html.Div(id='tabs-content')
])

#%%
@app.callback(Output('tabs-content', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return tab1_layout
    elif tab == 'tab-2':
        return tab2_layout
    elif tab == 'tab-3':
        return tab3_layout

#%%
if __name__ == '__main__':
    app.run_server(debug=True)














