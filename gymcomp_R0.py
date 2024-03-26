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
import dash_bootstrap_components as dbc
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

#%%  Function to calculate the color based on the score
def get_color(score, max_score):
    if math.isnan(score):
        return 'black'  # or any other default color
    else:
        # Calculate the color based on the score and max score
        color_value = score / max_score
        return color_value

#%% 
app = dash.Dash(__name__, suppress_callback_exceptions=True)

###################################
#%% Tab 1: Competition Overview ###
###################################
# Function to calculate the color based on the score
def get_color(score, max_score):
    if math.isnan(score):
        return 'black'  # or any other default color
    else:
        # Calculate the color based on the score and max score
        color_value = score / max_score
        return color_value

# Function to update the bubble plot
def update_bubble_plot(day, apparatus):
    data = {'x': [], 'y': [], 'size': [], 'name': [], 'score': [], 'color': []}
    max_score = max([stats['Score'] for values in database.values() if day in values for app, stats in values[day].items() if app == apparatus])

    exp = 3  # Adjust this as needed

    for name, values in database.items():
        if day in values:
            for app, stats in values[day].items():
                if app == apparatus:
                    if stats['E'] == 0.0:
                        data['x'].append(np.nan)
                    else:
                        data['x'].append(stats['E'])

                    if stats['D'] == 0.0:
                        data['y'].append(np.nan)
                    else:
                        data['y'].append(stats['D'])

                    data['name'].append(name)
                    data['score'].append(stats['Score'])
                    
                    # Make it zero if it's nan
                    if math.isnan(stats['Score']):
                        size = 0.0
                        color = 0.0
                    else:
                        size = stats['Score']
                        color = stats['Score']
                        
                    data['color'].append(get_color(color ** exp, max_score ** exp))
                        
                    size_exp = 1.5
                    if apparatus == "AA":
                        data['size'].append((size / 6) ** size_exp)
                    else:
                        data['size'].append(size ** size_exp)
    return data

def update_table(day, apparatus, selected_athlete=None):
    # Filter the database based on selected day and apparatus
    filtered_data = {name: stats for name, values in database.items() if day in values for app, stats in values[day].items() if app == apparatus}
    
    # Create DataFrame from filtered data
    df = pd.DataFrame.from_dict(filtered_data, orient='index')
    
    # Sort DataFrame by Score in descending order (if tie, sort by E score for now)
    df = df.sort_values(by=['Score', 'E'], ascending=[False, False])
    
    # Reset index to include Athlete name as a column
    df = df.reset_index().rename(columns={'index': 'Athlete name'})
    
    # Truncate score values to 3 decimal points (do not round)
    df['D score'] = df['D'].map('{:.3f}'.format)
    df['E score'] = df['E'].map('{:.3f}'.format)
    df['Score'] = df['Score'].map('{:.3f}'.format)
    
    # Add rank column
    df['Rank'] = df.index + 1
    
    # Reorder columns
    df = df[['Rank', 'Athlete name', 'D score', 'E score', 'Score']]
    
    # Generate HTML table with highlighted row if a selected athlete is provided
    table_rows = []
    for i in range(len(df)):
        row_data = df.iloc[i]
        background_color = 'yellow' if row_data['Athlete name'] == selected_athlete else 'white'
        table_row = html.Tr([html.Td(row_data[col], style={'background-color': background_color}) for col in df.columns])
        table_rows.append(table_row)
    
    table = html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df.columns])] +
        # Body
        table_rows
    )
    
    return table



#I want to make the drop down selectors take up less width
dropdown_style = {'width': '30%'}  # Adjust the width as needed

# Define layout of the app
overview_layout = html.Div([
    html.H3('Competition Overview'),
    dbc.Row([
        dbc.Col([
            html.Div("Competition Day:", style={'marginRight': '10px', 'verticalAlign': 'middle'}),
            dcc.Dropdown(
                id='day-dropdown',
                options=[{'label': day, 'value': day} for day in next(iter(database.values())).keys()],
                value=list(next(iter(database.values())).keys())[0],
                style=dropdown_style
            ),
        ], width=6),
        dbc.Col([
            html.Div("Apparatus:", style={'marginRight': '10px', 'verticalAlign': 'middle'}),
            dcc.Dropdown(
                id='apparatus-dropdown',
                options=[{'label': app, 'value': app} for app in ["FX", "PH", "SR", "VT", "PB", "HB", "AA"]],
                value='AA',
                style=dropdown_style
            ),
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='bubble-plot'),
            width=6
        ),
        dbc.Col(
            html.Div(id='table-container'),
            width=6
        )
    ])
])


# Define callback to update the bubble plot and table based on selected options
# Define callback to update the bubble plot and table based on selected options
@app.callback(
    [Output('bubble-plot', 'figure'),
     Output('table-container', 'children')],
    [Input('day-dropdown', 'value'),
     Input('apparatus-dropdown', 'value'),
     Input('bubble-plot', 'clickData')]  # Add clickData as input
)
def update_plot_and_table(day, apparatus, clickData):
    # Update bubble plot
    data = update_bubble_plot(day, apparatus)
    fig = px.scatter(data, x='x', y='y', color='color', size='size', hover_name='name',
                     color_continuous_scale='Viridis', opacity=0.6, hover_data={'name': True, 'x': False, 'y': False, 'size': False})
    fig.update_layout(title="Interactive Chart", 
                      xaxis_title="E score", 
                      yaxis_title="D score", 
                      autosize=True,
                      margin=dict(l=40, r=40, t=40, b=40),
                      width=1000, #play with this value until you like it
                      height=600,
                      )
    fig.update_traces(text=data['score'], textposition='top center')  

    # Customize hover template
    hover_template = ("<b>%{hovertext}</b><br>" +
                      "D score: %{y:.3f}<br>" +
                      "E score: %{x:.3f}<br>" +
                      "Score: %{text:.3f}")
    fig.update_traces(hovertemplate=hover_template)
    
    # Update color bar legend
    fig.update_coloraxes(colorbar_title="Score")
    
    # Map color values to score values for color bar tick labels
    color_values = np.linspace(0, 1, 11)  
    max_score = max(data['score'])
    score_values = [value * max_score for value in color_values]  
    
    # Update color bar tick labels
    fig.update_coloraxes(colorbar_tickvals=color_values, colorbar_ticktext=[f"{score:.3f}" for score in score_values])
    
    # If a point is clicked, highlight the corresponding row in the table
    if clickData:
        selected_athlete = clickData['points'][0]['hovertext']
        table = update_table(day, apparatus, selected_athlete)
    else:
        table = update_table(day, apparatus)
    
    return fig, table


########################################
#%% Tab 2: Individual Athlete Analysis #
########################################
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
        dcc.Tab(label='Competition Overview', value='tab-1'),
        dcc.Tab(label='Individual Athlete Analysis', value='tab-2'),
        dcc.Tab(label='Team Scenarios', value='tab-3'),
    ]),
    html.Div(id='tabs-content')
])

#%%
@app.callback(Output('tabs-content', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return overview_layout
    elif tab == 'tab-2':
        return tab2_layout
    elif tab == 'tab-3':
        return tab3_layout

#%%
if __name__ == '__main__':
    app.run_server(debug=True)














