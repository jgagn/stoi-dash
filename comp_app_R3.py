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

# Assuming you have your database variable defined with your data
database = {...}  # Your database variable

# Get the absolute path to the directory containing the main app file
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the file
file_path = os.path.join(base_dir, "test_data/EliteCanada2024/jrdatabase")

with open(file_path, 'rb') as f:
    database = pickle.load(f)

print("Database loaded successfully.")

# Create Dash app
app = dash.Dash(__name__)

# Define Athlete Analysis Layout
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
            {'label': 'E', 'value': 'E'},
            {'label': 'Summary', 'value': 'Summary'}
        ],
        value='D'  # Default value
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

    data = []

    if y_axis == 'Summary':
        # For 'Summary', create separate vertical bars for each day, with D and E scores stacked within each bar
        for result in selected_results:
            d_values = [database[selected_athlete][result].get(x, {'D': 0})['D'] for x in ["FX", "PH", "SR", "VT", "PB", "HB"]]
            e_values = [database[selected_athlete][result].get(x, {'E': 0})['E'] for x in ["FX", "PH", "SR", "VT", "PB", "HB"]]

            bar = go.Bar(
                x=["FX", "PH", "SR", "VT", "PB", "HB"], 
                y=[d + e for d, e in zip(d_values, e_values)],  # Sum of D and E scores
                name=result
            )

            data.append(bar)

        layout = go.Layout(
            title=f'{selected_athlete} - {", ".join(selected_results)} (Summary)',
            xaxis=dict(title='Apparatus'),
            yaxis=dict(title='Score')
        )
    else:
        # For individual days, create separate bars for each day
        for result in selected_results:
            d_values = [database[selected_athlete][result].get(x, {'D': 0})['D'] for x in ["FX", "PH", "SR", "VT", "PB", "HB"]]
            e_values = [database[selected_athlete][result].get(x, {'E': 0})['E'] for x in ["FX", "PH", "SR", "VT", "PB", "HB"]]

            trace_d = go.Bar(x=["FX", "PH", "SR", "VT", "PB", "HB"], y=d_values, name=f'{result} D')
            trace_e = go.Bar(x=["FX", "PH", "SR", "VT", "PB", "HB"], y=e_values, name=f'{result} E')

            data.extend([trace_d, trace_e])

        layout = go.Layout(
            title=f'{selected_athlete} - {", ".join(selected_results)} ({y_axis})',
            xaxis=dict(title='Apparatus'),
            yaxis=dict(title=y_axis)
        )

    return {'data': data, 'layout': layout}

# Define the entire layout with tabs
app.layout = html.Div([
    athlete_layout
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
