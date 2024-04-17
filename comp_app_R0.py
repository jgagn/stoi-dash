#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 13:36:23 2024

@author: joelgagnon
"""


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
#Now import informations for each tab seperately
# from athlete_layout import athlete_layout, update_result_dropdown, update_graph
# from athlete_layout import *
# Assuming you have your database variable defined with your data

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

#%% Define Competition Overview Layout

# Function to calculate the color based on the score
def get_color(score, max_score):
    if math.isnan(score):
        return 'black'  # or any other default color
    else:
        # Calculate the color based on the score and max score
        color_value = score / max_score
        return color_value

# Function to update the bubble plot
# def update_bubble_plot(day, apparatus):
#     data = {'x': [], 'y': [], 'size': [], 'name': [], 'score': [], 'color': []}
#     max_score = max([stats['Score'] for values in database.values() if day in values for app, stats in values[day].items() if app == apparatus])

#     exp = 3  # Adjust this as needed

#     for name, values in database.items():
#         if day in values:
#             for app, stats in values[day].items():
#                 if app == apparatus:
#                     if stats['E'] == 0.0:
#                         data['x'].append(np.nan)
#                     else:
#                         data['x'].append(stats['E'])

#                     if stats['D'] == 0.0:
#                         data['y'].append(np.nan)
#                     else:
#                         data['y'].append(stats['D'])

#                     data['name'].append(name)
#                     data['score'].append(stats['Score'])
                    
#                     #make it zero if its nan
#                     if math.isnan(stats['Score']):
#                         size = 0.0
#                         color = 0.0
#                     else:
#                         size = stats['Score']
#                         color = stats['Score']
                        
#                     data['color'].append(get_color(color ** exp, max_score ** exp))
                        
                        
#                     size_exp = 1.5
#                     if apparatus == "AA":
#                         data['size'].append((size/ 6) ** size_exp)
#                     else:
#                         data['size'].append(size ** size_exp)
                    
#     return data

#chatgpt says update bubble plot
def update_bubble_plot(day, apparatus):
    data = []
    max_score = max([stats['Score'] for values in database.values() if day in values for app, stats in values[day].items() if app == apparatus])

    exp = 3  # Adjust this as needed

    for name, values in database.items():
        if day in values:
            for app, stats in values[day].items():
                if app == apparatus:
                    if stats['E'] == 0.0:
                        x = None
                    else:
                        x = stats['E']

                    if stats['D'] == 0.0:
                        y = None
                    else:
                        y = stats['D']

                    color = get_color(stats['Score'] ** exp, max_score ** exp)

                    size_exp = 1.5
                    if apparatus == "AA":
                        size = (stats['Score'] / 6) ** size_exp
                    else:
                        size = stats['Score'] ** size_exp

                    data.append({'x': x, 'y': y, 'size': size, 'name': name, 'score': stats['Score'], 'color': color})
        print(f"debugging: data = {data}")
    return data



def generate_top_scores_table(day, apparatus, n=10):
    data = update_bubble_plot(day, apparatus)
    
    # Convert data into DataFrame
    df = pd.DataFrame(data)
    
    # Sort DataFrame by 'score' column in descending order
    df = df.sort_values(by='score', ascending=False)
    
    # Select top N rows
    df = df.head(n)
    
    # Define columns for the table
    columns = ['name', 'D', 'E', 'score']
    
    # Create Dash DataTable
    table = dash_table.DataTable(
        id='top-scores-table',
        columns=[{'name': col, 'id': col} for col in columns],
        data=df.to_dict('records'),
        style_table={'overflowX': 'auto'}
    )
    
    return table
#version below only makes the plot but the plot looks good

#New - with data table
# Define Competition Overview Layout
overview_layout = html.Div([
    dcc.Dropdown(
        id='day-dropdown',
        options=[{'label': day, 'value': day} for day in next(iter(database.values())).keys()],
        value=list(next(iter(database.values())).keys())[0]
    ),
    dcc.Dropdown(
        id='apparatus-dropdown',
        options=[{'label': app, 'value': app} for app in ["FX", "PH", "SR", "VT", "PB", "HB", "AA"]],
        value='AA'
    ),
    dcc.Graph(id='bubble-plot'),
    html.Div(id='top-scores-table')
])


# Define callback to update the bubble plot based on selected options
@app.callback(
    Output('bubble-plot', 'figure'),
    [Input('day-dropdown', 'value'),
      Input('apparatus-dropdown', 'value')]
)
def update_plot(day, apparatus):
    data = update_bubble_plot(day, apparatus)
    fig = px.scatter(data, x='x', y='y', color='color', size='size', hover_name='name', text='name', 
                      color_continuous_scale='Viridis', opacity=0.6)
    fig.update_layout(title="Interactive Chart", 
                      xaxis_title="E score", 
                      yaxis_title="D score", 
                      # aspectmode='manual',  # Set aspect mode to manual
                      # aspectratio=dict(x=1, y=1) ) # Set aspect ratio to 1:1 for a square plot
                      autosize=True,  # Automatically adjust the size based on the container
                      margin=dict(l=40, r=40, t=40, b=40)  # Add margins for better mobile display
                      )
    return fig


#trying chat gpt way

#New - with data table
# Define Competition Overview Layout

# overview_layout = html.Div([
#     dcc.Dropdown(
#         id='day-dropdown',
#         options=[{'label': day, 'value': day} for day in next(iter(database.values())).keys()],
#         value=list(next(iter(database.values())).keys())[0]
#     ),
#     dcc.Dropdown(
#         id='apparatus-dropdown',
#         options=[{'label': app, 'value': app} for app in ["FX", "PH", "SR", "VT", "PB", "HB", "AA"]],
#         value='AA'
#     ),
#     html.Div(id='bubble-plot-table-container')
# ])

# # Define callback to update the bubble plot and table based on selected options
# @app.callback(
#     Output('bubble-plot-table-container', 'children'),
#     [Input('day-dropdown', 'value'),
#      Input('apparatus-dropdown', 'value')]
# )
# def update_plot_and_table(day, apparatus):
#     bubble_plot_data = update_bubble_plot(day, apparatus)
#     bubble_plot_layout = {
#         'title': "Interactive Chart",
#         'xaxis_title': "E score",
#         'yaxis_title': "D score",
#         'autosize': True,
#         'margin': dict(l=40, r=40, t=40, b=40),
#     }
    
    
#     # bubble_plot = dcc.Graph(
#     #     id='bubble-plot',
#     #     figure={'data': bubble_plot_data, 'layout': bubble_plot_layout}
#     # )
    
    
#     fig = px.scatter(bubble_plot_data, x='x', y='y', color='color', size='size', hover_name='name', text='name', 
#                           color_continuous_scale='Viridis', opacity=0.6)
#     fig.update_layout(title="Interactive Chart", 
#                           xaxis_title="E score", 
#                           yaxis_title="D score", 
#                           # aspectmode='manual',  # Set aspect mode to manual
#                           # aspectratio=dict(x=1, y=1) ) # Set aspect ratio to 1:1 for a square plot
#                           autosize=True,  # Automatically adjust the size based on the container
#                           margin=dict(l=40, r=40, t=40, b=40)  # Add margins for better mobile display
#                           )
#     bubble_plot = fig
    
#     top_scores_table = generate_top_scores_table(day, apparatus)
    
#     return [bubble_plot, top_scores_table]  # Wrap in a list




#%% Define Athlete Analysis Layout
athlete_layout = html.Div([
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
#line graph

# @app.callback(
#     Output('individual-graph', 'figure'),
#     [Input('athlete-dropdown', 'value'),
#       Input('result-dropdown', 'value'),
#       Input('y-axis-dropdown', 'value')]
#     )


# def update_graph(selected_athlete, selected_results, y_axis):
#     if selected_athlete is None or selected_results is None:
#         # If no athlete or result selected, return an empty graph
#         return {'data': [], 'layout': {'title': 'Select athlete and results to display'}}

#     traces = []
#     for result in selected_results:
#         x_values = [key for key in database[selected_athlete][result].keys() if key != "AA"]
#         y_values = [database[selected_athlete][result][x][y_axis] for x in x_values]
#         # Filter out zeros
#         x_values = [x if x != 0.0 else np.nan for x in x_values]
#         y_values = [y if y != 0.0 else np.nan for y in y_values]

#         trace = go.Scatter(x=x_values, y=y_values, mode='lines+markers', name=result)
#         traces.append(trace)

#     layout = go.Layout(
#         title=f'{selected_athlete} - {", ".join(selected_results)} ({y_axis})',
#         xaxis=dict(title='Apparatus'),
#         yaxis=dict(title=y_axis),
#         legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
#     )

#     return {'data': traces, 'layout': layout}

#bar graph

Modify the update_graph callback function
@app.callback(
    Output('individual-graph', 'figure'),
    [Input('athlete-dropdown', 'value'),
      Input('result-dropdown', 'value'),
      Input('y-axis-dropdown', 'value')]
)
def update_graph(selected_athlete, selected_results, y_axis):
    if selected_athlete is None:
        # If no athlete selected, return an empty graph
        return {'data': [], 'layout': {'title': 'Select an athlete to display results'}}

    if selected_results is None:
        # If no specific results selected, use day1 and day2 as initial results
        selected_results = ['day1', 'day2']

    data = []
    for result in selected_results:
        x_values = [key for key in database[selected_athlete][result].keys() if key != "AA"]
        y_values = [database[selected_athlete][result][x][y_axis] for x in x_values]
        
        # Filter out zeros
        x_values = [x if x != 0.0 else np.nan for x in x_values]
        y_values = [y if y != 0.0 else np.nan for y in y_values]
        
        # Create a bar trace
        trace = go.Bar(x=x_values, y=y_values, name=result)
        data.append(trace)

    layout = go.Layout(
        title=f'{selected_athlete} - {", ".join(selected_results)} ({y_axis})',
        xaxis=dict(title='Apparatus'),
        yaxis=dict(title=y_axis),
        barmode='group'  # Display bars grouped
    )

    return {'data': data, 'layout': layout}

#try stacking - didnt work

# @app.callback(
#     Output('individual-graph', 'figure'),
#     [Input('athlete-dropdown', 'value'),
#      Input('result-dropdown', 'value'),
#      Input('y-axis-dropdown', 'value')]
# )

# def update_graph(selected_athlete, selected_results, y_axis):
#     if selected_athlete is None or selected_results is None:
#         # If no athlete or result selected, return an empty graph
#         return {'data': [], 'layout': {'title': 'Select athlete and results to display'}}

#     if y_axis == 'Summary':
#         # For 'Summary', calculate the sum of D and E scores to create stacked bars
#         data = []
#         for result in selected_results:
#             d_values = [database[selected_athlete][result][x]['D'] for x in database[selected_athlete][result]]
#             e_values = [database[selected_athlete][result][x]['E'] for x in database[selected_athlete][result]]
#             total_values = np.array(d_values) + np.array(e_values)

#             trace_d = go.Bar(x=list(database[selected_athlete][result].keys()), y=d_values, name='D Score')
#             trace_e = go.Bar(x=list(database[selected_athlete][result].keys()), y=e_values, name='E Score')
#             trace_total = go.Bar(x=list(database[selected_athlete][result].keys()), y=total_values, name='Total Score')

#             data.extend([trace_d, trace_e, trace_total])

#         layout = go.Layout(
#             title=f'{selected_athlete} - {", ".join(selected_results)} (Summary)',
#             xaxis=dict(title='Apparatus'),
#             yaxis=dict(title='Score'),
#             barmode='stack'  # Stack bars for D and E scores
#         )
#     else:
#         # For other options, create individual bars
#         data = []
#         for result in selected_results:
#             y_values = [database[selected_athlete][result][x][y_axis] for x in database[selected_athlete][result]]
#             trace = go.Bar(x=list(database[selected_athlete][result].keys()), y=y_values, name=result)
#             data.append(trace)

#         layout = go.Layout(
#             title=f'{selected_athlete} - {", ".join(selected_results)} ({y_axis})',
#             xaxis=dict(title='Apparatus'),
#             yaxis=dict(title=y_axis),
#             barmode='group'  # Display bars grouped
#         )

#     return {'data': data, 'layout': layout}
#%% Assemble all tabs into app

#blank first tab (placeholder)
# Define the layout for the second tab
tab1_layout = overview_layout

#Second Tab is the Athlete Layout
tab2_layout = athlete_layout

# Define the entire layout with tabs
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Competition Overview', value='tab-1', children=tab1_layout),
        dcc.Tab(label='Athlete Analysis', value='tab-2', children=tab2_layout),
    ])
])


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)