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
from dash import dcc, html, dash_table, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import math
import plotly.graph_objs as go
import plotly.express as px
import pickle
import os
import itertools
from plotly.subplots import make_subplots
#dash authentication
# import dash_auth # pip install dash-auth==2.0.0. <- add this to requirements.txt

# Import the print function
from builtins import print

#import the team scenarios calc
from team_scenario_calcs import team_score_calcs

#ordered dict
from collections import OrderedDict
#%% Import Data 
#use absolute path

# Get the absolute path to the directory containing the main app file
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the file
#file path and csv file name
path = "test_data/Combined"
pkl_file = "gymcan_mag_athletes"
file_path = os.path.join(base_dir, path+"/"+pkl_file)


with open(file_path, 'rb') as f:
    database = pickle.load(f)

# print("Database loaded successfully.")

#%%  Function to calculate the color based on the score
def get_color(score, max_score):
    if math.isnan(score):
        return 'black'  # or any other default color
    else:
        # Calculate the color based on the score and max score
        color_value = score / max_score
        return color_value

#%% Setup App, Title, Authentication
app = dash.Dash(__name__, suppress_callback_exceptions=True) #, external_stylesheets=[dbc.themes.MORPH])

app.title = "STOI Demo"

# # Keep this out of source code repository - save in a file or a database
# VALID_USERNAME_PASSWORD_PAIRS = {
#     'hello3': 'world'
# }

# #ssecret key for flask
# app.secret_key = 'mensartisticgymnasticsdemo'  # Set your secret key here

# auth = dash_auth.BasicAuth(
#     app,
#     VALID_USERNAME_PASSWORD_PAIRS
# )

#%% Helpful functions

def get_category_data_for_competition_day(database, competition, categories, results, apparatus):
    #Since we made category a multi-select, we need to update how the code works
    #categories is now a list
    
    data = {}
    
    for athlete, competitions in database.items():
        
        if athlete not in {"overview", "competition_acronyms", "category_acronyms"}:
            # print(f"athlete: {athlete}")
            # print(f"competition: {competition}")
            # print(f"competitions: {competitions.keys()}")
            if competition in competitions.keys():
                # print(f"athlete: {athlete}")
                #loop through categories if they are selected
                if categories:
                    for category in categories:
                        if category == database[athlete][competition]['category']:
                            #need to make sure we have selected results, it might be none
                            if results != None:
                                try:
                                    data[athlete] = database[athlete][competition][results][apparatus]
                                    #I also want to add the category of the athlete into the data
                                    #this is because we have multi-select categories now and it is now useful to know
                                    data[athlete]['category'] = database[athlete][competition]['category']
                                    
                                except:
                                    #There are some scenarios where an athlete only competes day 1 but not day 2 or vice versa
                                    #also important for finals
                                    #in those cases, if we cant find the data,do not try to set it as it does not exist
                            
                                    # print(f"couldn't save {athlete} data for {results}")
                                    pass
                                    
    # print(data)
    return data

########################
#%% Global Variables ###
########################

#I want to make the drop down selectors take up less width
dropdown_style = {'width': '50%'}  # Adjust the width as needed


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
def update_bubble_plot(database, competition, categories, results, apparatus):
    data = {'x': [], 'y': [], 'category': [], 'size': [], 'name': [], 'score': [], 'color': []}
    
    #filter the data 

    bubble_data = get_category_data_for_competition_day(database, competition, categories, results, apparatus)
    
    if not bubble_data:
        # print("no bubble plot data")
        # table = html.Table()
        pass
    else:
        # print("we have bubble data!")
        # print(bubble_data)
        
        #TODO change this max score thing likely
        max_score = max([values['Score'] for values in bubble_data.values()])
        # max_score = 16
        # print(f"max score: {max_score}")
        exp = 3  # Adjust this as needed
        
        
        for name, stats in bubble_data.items():
            # print(f"name: {name}")
            # print(f"stats: {stats}")
            #I've already filtered the apparatus
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
                
            #add category data
            data['category'].append(stats['category'])
    return data

def update_table(database, competition, categories, results, apparatus, selected_athlete=None):
    # Filter the database based on selected day and apparatus
    # filtered_data = {name: stats for name, values in database.items() if day in values for app, stats in values[day].items() if app == apparatus}
    
    
    
    table_data = get_category_data_for_competition_day(database, competition, categories, results, apparatus)
    
    # Ensure that the table_data dictionary is not empty
    if not table_data:
        # print("no table data")
        table = html.Table()
    else:
        # print("we have table data!")
        # Flatten the dictionary and convert to DataFrame
        df = pd.DataFrame.from_dict(table_data, orient='index')
        
        # Check if the DataFrame has the expected columns
        expected_columns = ['Score', 'E']  # Add other expected columns here
        if not set(expected_columns).issubset(df.columns):
            return None
        
        # Sort DataFrame by Score in descending order (if tie, sort by E score for now)
        df = df.sort_values(by=['Score', 'E'], ascending=[False, False])
        
        # print(f"df: {df}")
        # Reset index to include Athlete name as a column
        df = df.reset_index().rename(columns={'index': 'Athlete name'})
        
        #Fill any nans to 0.000
        df = df.fillna(0.000)
        
        # Truncate score values to 3 decimal points (do not round)
        df['D score'] = df['D'].map('{:.3f}'.format)
        df['E score'] = df['E'].map('{:.3f}'.format)
        df['Score'] = df['Score'].map('{:.3f}'.format)
        
        # create "Category" column with capital "C" and map the acronyms to the full text
        df['Category'] = df['category'].map(database['category_acronyms'])
        
        # Add rank column
        df['Rank'] = df.index + 1
        
        # Reorder columns
        df = df[['Rank', 'Athlete name', 'Category','D score', 'E score', 'Score']]
        
        # Generate HTML table with highlighted row if a selected athlete is provided
        table_rows = []
        for i in range(len(df)):
            row_data = df.iloc[i]
            background_color = 'yellow' if row_data['Athlete name'] == selected_athlete else ('white' if i % 2 == 0 else '#e6f2ff') #making it blue if not selected
            table_row = html.Tr(
                [html.Td(row_data[col], style={'background-color': background_color, 'padding': '10px'}) for col in df.columns]
            )
            table_rows.append(table_row)
        
        table = html.Table(
            # # Header
            # [html.Tr([html.Th(col) for col in df.columns])] +
            # # Body
            # table_rows
            # Header
            # Header
            [html.Tr([html.Th(col, style={'padding': '10px', 'background-color': '#cce7ff'}) for col in df.columns])] +
            # Body
            table_rows,
            style={'border-collapse': 'collapse', 'width': 'auto'}
            
        )
    
    return table

# Define layout of the app

#I might want to make Cateogry, and other drop downs, multi-select

overview_layout = html.Div([
    html.H3('Competition Data Selection'),
    dbc.Row([
        dbc.Col([
            html.Div("Competition", style={'marginRight': '10px', 'verticalAlign': 'middle'}),
            dcc.Dropdown(
                id='competition-dropdown',
                options=[{'label': database['competition_acronyms'][comp], 'value': comp} for comp in database['overview'].keys()],
                value=list(next(iter(database.values())).keys())[0],
                style=dropdown_style
            ),
        ], width=6),
        dbc.Col([
            html.Div("Category (can select more than 1):", style={'marginRight': '10px', 'verticalAlign': 'middle'}),
            dcc.Dropdown(
                id='category-dropdown',
                style=dropdown_style,
                multi=True  # Enable multi-select
            ),
        ], width=6),
        dbc.Col([
            html.Div("Results:", style={'marginRight': '10px', 'verticalAlign': 'middle'}),
            dcc.Dropdown(
                id='results-dropdown',
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
    dcc.Store(id='results-store', data=database),  # Store the database - needed to dynamically change data in dropdown menus
    
    
    dbc.Row([
        html.H3('Interactive Bubble Plot'),
        dbc.Col(
            dcc.Graph(id='bubble-plot'),
            width=6
        ),
        html.H3("Data Table"),
        
        dbc.Col(
            html.Div(id='table-container'),
            width=6
        )
    ])
])

# Define callback to update the options of the results dropdown based on the selected competition and category
@app.callback(
    Output('results-dropdown', 'options'),
    [Input('competition-dropdown', 'value'),
     Input('category-dropdown', 'value')],
    [State('results-store', 'data')]
)
def update_results_dropdown(competition, categories, database):
    # print("Competition:", competition)
    # print("Categories:", categories)
    # print("Database:", database)
    
    #category is now a multi-select option
    #will need to only show the results options that correspond to multi categories
    if competition and categories:
        results_options = []
        #although categories should be a list, sometimes it returns just one
        #lets make sure it is a list if its not
        if not isinstance(categories, list):
            categories = [categories]
            
        for category in categories:
            # print(f"category: {category}")
            # Get the available results options from the database dictionary
            options = database['overview'][competition][category] 
            results_options.append(options)
        #now, only keep the options that show up for all categories
        # print(f"result_options: {results_options}")
        # Create options for the results dropdown
        
    
        def find_common_elements(list_of_lists):
            # Convert each sublist to a set
            sets = [set(sublist) for sublist in list_of_lists]
            
            # Find the intersection of all sets
            common_elements = set.intersection(*sets)
            
            # Convert the result back to a list
            return sorted(common_elements)
        
        common_elements = find_common_elements(results_options)
        # print(common_elements)
                
        
        return [{'label': result, 'value': result} for result in common_elements + ["average","best"]]
    else:
        return []

# Define callback to set the value of the results dropdown to the first option when the competition or category changes
@app.callback(
    Output('results-dropdown', 'value'),
    [Input('competition-dropdown', 'value'),
     Input('category-dropdown', 'value')],
    [State('results-dropdown', 'options')]
)
def set_results_dropdown_value(competition, category, options):
    if options:
        return options[0]['value']
    else:
        return None

# Define callback to update the options of the category dropdown based on the selected competition
@app.callback(
    Output('category-dropdown', 'options'),
    [Input('competition-dropdown', 'value')],
    [State('results-store', 'data')]
)
def update_category_dropdown(competition, database):
    if competition:
        category_options = database['overview'][competition].keys()
        # Create options for the results dropdown
        return [{'label': database['category_acronyms'][category], 'value': category} for category in category_options]
    else:
        return []

# Define callback to set the value of the category dropdown to the first option when the competition changes
@app.callback(
    Output('category-dropdown', 'value'),
    [Input('competition-dropdown', 'value')],
    [State('category-dropdown', 'options')]
)
def set_category_dropdown_value(competition, options):
    if options:
        return options[0]['value']
    else:
        return None


# Define callback to update the bubble plot and table based on selected options

@app.callback(
    [Output('bubble-plot', 'figure'),
     Output('table-container', 'children')],
    [Input('results-dropdown', 'value'),
     Input('apparatus-dropdown', 'value'),
     Input('category-dropdown', 'value'),
     Input('competition-dropdown', 'value'),
     Input('bubble-plot', 'clickData')]  # Add clickData as input
)
def update_plot_and_table(results, apparatus, categories, competition, clickData):
    # Update bubble plot
    # print(f"plot and table categories: {categories}")
    
    #need to make sure categories is a list, it should  be sometimes isn't
    if not isinstance(categories, list):
        categories = [categories]
    
    data = update_bubble_plot(database, competition, categories, results, apparatus)
    
    #Adding full category name in the data
    #ordered dict seems to be needed to make sure when i convert from ar=cronym to name the order stays the same
    mapped_data = OrderedDict()
    
    if data['x']:
        for key, values in data.items():
            if key == 'category':
                mapped_data['category'] = []
                for value in values:
                    mapped_data['category'].append(database['category_acronyms'].get(value, value))
            else:
                mapped_data[key] = values
    
        # print(f"mapped_data: {mapped_data}")
    
        # Let's use this new mapped data!
        data = mapped_data
        # print(f"data: {data}")
        
    fig = px.scatter(data, x='x', y='y', color='color', size='size', hover_name='name',
                     color_continuous_scale='Viridis', opacity=0.6, hover_data={'name': True,'category':True, 'x': False, 'y': False, 'size': False})
    fig.update_layout(title=f"{database['competition_acronyms'][competition]}: D score vs. E score", 
                      xaxis_title="E score", 
                      yaxis_title="D score", 
                      autosize=True,
                      margin=dict(l=40, r=40, t=40, b=40),
                      width=1000, #play with this value until you like it
                      height=600,
                      )
    fig.update_traces(text=data['score'], textposition='top center')  

    # Customize hover template
    hover_template = (
        "<b>%{hovertext}</b><br>" +
        "Category: %{customdata}<br>" +
        "D score: %{y:.3f}<br>" +
        "E score: %{x:.3f}<br>" +
        "Score: %{text:.3f}"
    )
    
    fig.update_traces(hovertemplate=hover_template, customdata=data['category'])
    
    # Update color bar legend
    fig.update_coloraxes(colorbar_title="Score")
    
    # Map color values to score values for color bar tick labels
    color_values = np.linspace(0, 1, 11)  
    
    #only try to get a max score if we have plottable data, otherwise set max_score to an arbitrary value
    if not data['x']:
        max_score = 16
    else:
        max_score = np.nanmax(data['score'])
        # print(f"max score: {max_score}")
        score_values = [value * max_score for value in color_values]  
        # print(f"score values: {score_values}")
        
        # Update color bar tick labels
        fig.update_coloraxes(colorbar_tickvals=color_values, colorbar_ticktext=[f"{score:.3f}" for score in score_values])
    
    # If a point is clicked, highlight the corresponding row in the table
    if clickData:
        selected_athlete = clickData['points'][0]['hovertext']
        table = update_table(database, competition, categories, results, apparatus, selected_athlete)
    else:
        table = update_table(database, competition, categories, results, apparatus)
    
    return fig, table


########################################
#%% Tab 2: Individual Athlete Analysis #
########################################

#colour dictionary

barplot_colours = {'D':
                       ['rgb(31, 119, 180)',  # Blue
                        'rgb(255, 127, 14)',  # Orange
                        'rgb(44, 160, 44)',   # Green
                        'rgb(214, 39, 40)',   # Red
                        'rgb(148, 103, 189)', # Purple
                        'rgb(140, 86, 75)',   # Brown
                        'rgb(227, 119, 194)'  # Pink
                        ]
                        ,
                   'E':
                       ['rgba(31, 119, 180, 0.5)',  # Light Blue
                        'rgba(255, 127, 14, 0.5)',  # Light Orange
                        'rgba(44, 160, 44, 0.5)',   # Light Green
                        'rgba(214, 39, 40, 0.5)',   # Light Red
                        'rgba(148, 103, 189, 0.5)', # Light Purple
                        'rgba(140, 86, 75, 0.5)',   # Light Brown
                        'rgba(227, 119, 194, 0.5)'  # Light Pink
                        ]
                     }
def barplot_width(n):
    if n == 1:
        width = 0.4
    elif n ==2:
        width = 0.3
    elif n == 3:
        width = 0.225
    elif n==4:
        width = 0.175
    elif n==5:
        width = 0.175/2.2
    elif n==5:
        width = 0.175/4.5
    else:
        width=0.0
    return width

# Define layout for the second tab with dropdowns and bar graph
exclude_keys = ["overview", "competition_acronyms", "category_acronyms"]

#SUBPLOT CODE

# Define function to generate subplot based on athlete dropdown selection change
def generate_subplot(athlete):
    # Dummy data (replace with your actual data)
    competitions = ['Comp1', 'Comp2', 'Comp3', 'Comp4', 'Comp5']
    tlas = ['FX', 'PH', 'SR', 'VT', 'PB', 'HB', 'AA']
    scores = {
        'FX': [10, 20, 15, 25, 30],
        'PH': [15, 25, 20, 30, 35],
        'SR': [20, 30, 25, 35, 40],
        'VT': [25, 35, 30, 40, 45],
        'PB': [30, 40, 35, 45, 50],
        'HB': [35, 45, 40, 50, 55],
        'AA': [40, 50, 45, 55, 60]
    }

    # Create traces for each TLA
    traces = []
    for tla in tlas:
        trace = go.Scatter(
            x=competitions,
            y=scores[tla],
            mode='lines+markers',
            name=tla,
            # name=None,
        )
        traces.append(trace)

    # Create subplot with independent y-axes
    fig = make_subplots(rows=7, cols=1, shared_xaxes=True) #, subplot_titles=tlas)

    # Add traces to subplot
    for i, trace in enumerate(traces):
        fig.add_trace(trace, row=i + 1, col=1)

    # Update layout settings
    fig.update_layout(
        title='Trends Across Competitions',
        xaxis=dict(title='Competitions'),
        width=1000,
        height=1400,
        showlegend=False,
        margin=dict(l=40, r=40, t=0, b=0)  # Adjust the margins as needed
    )
    
    # Remove the x-axis title for the first subplot
    fig.update_xaxes(title='', row=1, col=1)
    fig.update_xaxes(title='Competitions', row=7, col=1)
    
    # add x-axis and y axis labels 
    for i in range(1, 8):
        fig.update_xaxes(showticklabels=True, row=i, col=1)
        fig.update_yaxes(title=tlas[i-1], row=i, col=1)
    return fig



tab2_layout = html.Div([
    html.H3('Specific Competition Apparatus Overview'),
    html.Div([
        html.Div("Athlete", style={'marginRight': '10px', 'verticalAlign': 'middle'}),
        dcc.Dropdown(
            id='athlete-dropdown2',
            options = [{'label': athlete, 'value': athlete} for athlete in database.keys() if athlete not in exclude_keys],
            value=next(iter(database)),  # Default value
            multi=False,  # Single select
            style=dropdown_style
        ),
        
        html.Div("Competition", style={'marginRight': '10px', 'verticalAlign': 'middle'}),
        dcc.Dropdown(
            id='competition-dropdown2',
            # options=[{'label': database['competition_acronyms'][comp], 'value': comp} for comp in database['overview'].keys()],
            value=list(next(iter(database.values())).keys())[0],
            style=dropdown_style
        ),
        
        html.Div("Results (can select more than 1):", style={'marginRight': '10px', 'verticalAlign': 'middle'}),
        dcc.Dropdown(
            id='results-dropdown2',
            # options=[{'label': day, 'value': day} for day in database[next(iter(database))].keys()],
            value=['day1','day2'],  # Default value
            multi=True,  # Allow multi-select
            style=dropdown_style
        )
    ]),
    dcc.Store(id='results-store2', data=database),  # Store the database - needed to dynamically change data in dropdown menus
    dcc.Graph(id='score-graph', style={'width': '1000px', 'height': '400px'}),
    html.H3('Trends Across Competions'),
    # Subplot will be added here based on athlete dropdown selection change
    dcc.Graph(id='subplot')
])

#SINGLE DROPDOWN CALLBACKS

# Define a single callback to update both competition and results dropdowns
@app.callback(
    [Output('competition-dropdown2', 'options'),
     Output('competition-dropdown2', 'value'),
     Output('results-dropdown2', 'options'),
     Output('results-dropdown2', 'value')],
    [Input('athlete-dropdown2', 'value'),
     Input('competition-dropdown2', 'value')],
    [State('results-store2', 'data')]
)
def update_dropdowns(athlete, competition, database):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'athlete-dropdown2':
        if athlete:
            competitions = list(database[athlete].keys())
            comp_options = [{'label': database['competition_acronyms'][comp], 'value': comp} for comp in competitions]
            
            return comp_options, None, [], None
        else:
            return [], None, [], None

    elif triggered_id == 'competition-dropdown2':
        if athlete and competition:
            results_options = [day for day in database[athlete][competition].keys() if day != "category"]
            return dash.no_update, dash.no_update, [{'label': result, 'value': result} for result in results_options], None
        else:
            return dash.no_update, dash.no_update, [], None

    return [], None, [], None
#PLOT 1 CALLBACKS

@app.callback(
    Output('score-graph', 'figure'),
    [Input('athlete-dropdown2', 'value'),
     Input('competition-dropdown2', 'value'),
     Input('results-dropdown2', 'value'),
     ]
)
def update_score_graph(athlete, competition, results):
    traces = []
    max_score = 0

    #lets check to see if we have everythin selected
    if athlete and competition and results:
        
        #lets make sure it is a list if its not make it one
        if not isinstance(results, list):
            results = [results]
        
        #width and offset will be based on number of days selected
        n_days = len(results)
        # print(f"n_days: {n_days}")
        
        # print(f"width: {barplot_width(n_days)}")
        width = barplot_width(n_days)
        
        # Define an offset multiplier for each day
        #starting with negative offset so we are always around zero
        offset_multiplier = -width*(n_days-1)/2
        
        for i,result in enumerate(results):
            # athlete = database[athlete][competition]
            d_scores = []
            e_scores = []
            plot_apparatus = ['FX','PH','SR','VT','PB','HB']
            
            for app in plot_apparatus:
                d_scores.append(database[athlete][competition][result][app]['D'])
                e_scores.append(database[athlete][competition][result][app]['E'])
                max_score = 16 #max(max_score, max(athlete[day][app]['Score'])) #+athlete[day][app]['E'])) #, athlete[day][app]['E']))
            
            # print(barplot_colours['D'][day])
            
            # Create stacked bar trace for D and E scores
            stacked_trace_d = go.Bar(
                x=[i + offset_multiplier for i in range(len(plot_apparatus))],  # Adjust x-location based on offset_multiplier
                y=d_scores,
                name=f'D score ({result})',
                # hoverinfo='y+name',
                hovertext=[f'{d:.3f}' for d in d_scores],
                hoverinfo='text+name',  # Use custom hover text and show trace name
                
                marker_color=barplot_colours['D'][i],  # Set color for D scores
                # marker_pattern='cross',
                # marker=dict(pattern='+', pattern_fgcolor='black'),
                # marker_pattern_fgcolor=barplot_colours['E'][day],
                offsetgroup=result,  # Group by day
                legendgroup=result,  # Group by day
                width = width,
            )
            
            
            stacked_trace_e = go.Bar(
                x=[i + offset_multiplier for i in range(len(plot_apparatus))],  # Adjust x-location based on offset_multiplier
                y=e_scores,
                name=f'E score ({result})',
                #custom hover text
                # hoverinfo='y+name',
                hovertext=[f'{e:.3f}' for e in e_scores],
                hoverinfo='text+name',  # Use custom hover text and show trace name
                
                marker_color=barplot_colours['E'][i],  # Set color for E scores
                offsetgroup=result,  # Group by day
                legendgroup=result,  # Group by day
                base=d_scores,  # Offset by D scores
                width = width,
                # Adding text above the bar plot for the whole score, truncated to 3 decimal places
                text=[f'{d + e:.3f}' for d, e in zip(d_scores, e_scores)],
                textposition='outside'
            )
            
            traces.append(stacked_trace_d)
            traces.append(stacked_trace_e)
            
            # Increment the offset multiplier for the next day
            offset_multiplier += width # Adjust the multiplier as needed to prevent overlapping bars
        
        layout = go.Layout(
        title=f'Score Breakdown for {athlete} at {database["competition_acronyms"][competition]}',
        xaxis={'title': 'Apparatus'},
        yaxis={'title': 'Score', 'range': [0, max_score * 1.1]},
        barmode='relative',  # Relative bars for stacked and grouped
        width=1000,
        height=400
        )
    
        # Set x-axis tick labels to be the apparatus names
        layout['xaxis']['tickvals'] = list(range(len(plot_apparatus)))
        layout['xaxis']['ticktext'] = plot_apparatus
        # print(f"plot app: {plot_apparatus}")
    else:
        #if we havent selected valid inputs yet, return blank
        traces = [go.Bar()]
        layout = go.Layout()
        
    return {'data': traces, 'layout': layout}


#SUBPLOT CALLBACKS

# Callback to update the subplot based on athlete dropdown selection change
@app.callback(
    Output('subplot', 'figure'),
    [Input('athlete-dropdown2', 'value')]
)
def update_subplot(athlete):
    return generate_subplot(athlete)


######################
#%% Team Scenarios ###
######################

# Sample data for demonstration
team_score_dummy = [
    {'Athlete': 'VANOUNOU Liam', 'FX': 12.475, 'PH': 12.025, 'SR': 12.125, 'VT': 12.900500000000001, 'PB': 12.45, 'HB': 11.8, 'Total': 73.7755},
    {'Athlete': 'GONZALEZ Aiden', 'FX': 12.725, 'PH': 11.625, 'SR': 11.8, 'VT': 12.466999999999999, 'PB': 11.274999999999999, 'HB': 12.925, 'Total': 72.81700000000001},
    {'Athlete': 'HUBER Evan', 'FX': 12.725, 'PH': 10.4, 'SR': 12.4, 'VT': 13.767, 'PB': 12.425, 'HB': 12.075, 'Total': 73.792},
    {'Athlete': 'MADORE Raphael', 'FX': 13.075, 'PH': 10.1, 'SR': 12.524999999999999, 'VT': 13.9505, 'PB': 12.625, 'HB': 11.875, 'Total': 74.1505},
    {'Athlete': 'CARROLL Jordan', 'FX': 0.0, 'PH': 13.625, 'SR': 0.0, 'VT': 0.0, 'PB': 0.0, 'HB': 0.0, 'Total': 13.625},
    {'Athlete': 'Team', 'FX': 38.525, 'PH': 37.275, 'SR': 37.05, 'VT': 40.618, 'PB': 37.5, 'HB': 36.875, 'Total': 227.843}
]


# Header for the table
header = ['Athlete', 'FX', 'PH', 'SR', 'VT', 'PB', 'HB', 'AA']

def generate_table(data):
    return dash_table.DataTable(
        columns=[{'name': i, 'id': i} for i in header],
        data=data,
        style_cell={'textAlign': 'center', 'whiteSpace': 'normal', 'height': 'auto'},  # Ensure text wraps within cells
        style_table={'overflowX': 'auto'},  # Enable horizontal scroll if content overflows
    )

tab3_layout = html.Div([
    html.Label('Results:'),
    dcc.Dropdown(
        id='results-dropdown',
        options=[
            {'label': 'Day 1', 'value': 'day1'},
            {'label': 'Day 2', 'value': 'day2'},
            {'label': 'Average', 'value': 'average'},
            {'label': 'Best', 'value': 'best'}
        ],
        value='day1',
        style={'width': '200px'}  # Adjust width here
    ),

    html.Label('Competition Format:'),
    html.Div([
        dcc.Input(id='xx-input', type='number', min=1, max=15, value=5, style={'width': '50px', 'fontSize': '16px'}),
        html.Label('-', style={'padding': '0 5px'}),  # Added label with padding
        dcc.Input(id='yy-input', type='number', min=1, max=6, value=4, style={'width': '50px',  'fontSize': '16px'}),
        html.Label('-', style={'padding': '0 5px'}),  # Added label with padding
        dcc.Input(id='zz-input', type='number', min=1, max=5, value=3, style={'width': '50px', 'fontSize': '16px'}),
    ]),
    
    html.Label('Show Top', style={'margin-top': '10px'}),  # Added label for the top X team scenarios
    dcc.Input(id='top-x-input', type='number', min=1, max=20, value=5, style={'width': '50px', 'fontSize': '16px'}),  # Added input box for X
    html.Label('team scenarios', style={'margin-left': '5px', 'margin-top': '10px'}),  # Added label for team scenario
    
    
    # html.Div(id="buttons-container", children=[
    #     html.Button('Calculate', id='calculate-button', n_clicks=0, style={'display': 'block', 'width': '150px', 'height': '40px', 'background-color': 'green', 'color': 'white', 'border': 'none', 'border-radius': '5px', 'fontSize': '20px'}),
    #     html.Button('Calculating...', id='loading-button', disabled=True, style={'display': 'none', 'width': '150px', 'height': '40px', 'background-color': 'gray', 'color': 'white', 'border': 'none', 'border-radius': '5px', 'fontSize': '20px'}),
    # ]),
    
    html.Button('Calculate', id='calculate-button', n_clicks=0, style={'display': 'block', 'margin-top': '10px', 'width': '150px', 'height': '40px', 'background-color': 'green', 'color': 'white', 'border': 'none', 'border-radius': '5px', 'fontSize': '20px'}),
    
    # # Loading button widget
    # dcc.Loading(
    #     id="loading-calculate",
    #     type="default",
    #     children=[
    #         html.Button('Calculating...', id='loading-button', disabled=True, style={'display': 'block', 'margin-top': '10px', 'width': '150px', 'height': '40px', 'background-color': 'gray', 'color': 'white', 'border': 'none', 'border-radius': '5px', 'fontSize': '20px'}),
    #         html.Div(id="loading-output-calculate")
    #     ]
    # ),
    
    # html.Button('Calculate', id='calculate-button', n_clicks=0, style={
    #     'display': 'block',
    #     'margin-top': '10px',
    #     'width': '150px',
    #     'height': '40px',
    #     'background-color': 'green',
    #     'color': 'white',
    #     'border': 'none',
    #     'border-radius': '5px',
    #     'fontSize': '20px',
    #     'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.2)',  # Add box shadow effect
    # }),
    
    # Placeholder for tables that will be updated based on filters
    html.Div(id='tables-container')
])

# Callback to switch between Calculate and Calculating buttons
# @app.callback(
#     Output('calculate-button', 'style'),
#     Output('loading-button', 'style'),
#     Input('calculate-button', 'n_clicks'),
#     prevent_initial_call=True
# )
# def toggle_buttons(n_clicks):
#     if n_clicks % 2 == 1:
#         return {'display': 'none'}, {'display': 'block'}
#     else:
#         return {'display': 'block'}, {'display': 'none'}

# @app.callback(
#     Output('calculate-button', 'n_clicks'),
#     Input('calculate-button', 'n_clicks'),
#     prevent_initial_call=True
# )
# def reset_n_clicks(n_clicks):
#     return 0

# Callback to generate tables when the "Calculate" button is clicked
@app.callback(
    Output('tables-container', 'children'),
    [Input('calculate-button', 'n_clicks')],
    [State('results-dropdown', 'value'),
      State('xx-input', 'value'),
      State('yy-input', 'value'),
      State('zz-input', 'value'),
      State('top-x-input', 'value')]
    )

def generate_tables(n_clicks, results_value, xx_value, yy_value, zz_value, num_scenarios):

    tables = []
    if n_clicks:
        
        #Here is where we will actually do the team score calculations!
        
        results_value = dash.callback_context.states['results-dropdown.value']
        xx_value = dash.callback_context.states['xx-input.value']
        yy_value = dash.callback_context.states['yy-input.value']
        zz_value = dash.callback_context.states['zz-input.value']
        
        
        
        comp_format = [xx_value,yy_value,zz_value]
        team_size = xx_value
        #get names
        names = []
        for name in database:
            names.append(name)

        #Here is where we could add the feature to NOT INCLUDE certain gymnasts
        # names.remove('CARROLL Jordan')
        # names.remove('ALLAIRE Dominic')
        
        all_combos = list(itertools.combinations(names, team_size))

        #let's try
        combo_scores = []
        # start_time = time.monotonic()
        for combo in all_combos:
            team_score = team_score_calcs(comp_format,combo,database,results=results_value,print_table=False)
            combo_scores.append(team_score['Team']['AA'])
        # end_time = time.monotonic()
        # time_for_all = timedelta(seconds=end_time - start_time)
        # print(f"Time for all: {time_for_all}")

        #% Try zipping lists and then sorting to rank
        combined = list(zip(list(combo_scores), all_combos))
        #sort it
        combined.sort(key=lambda x:x[0],reverse=True)
        #https://stackoverflow.com/questions/20099669/sort-multidimensional-array-based-on-2nd-element-of-the-subarray

        #colour coding if we want (TODO)
        colour_dict = {"scratch":"red",
                       "dropped":"black",
                       "counting":"white"}
        
        for i in range(num_scenarios):
            
            # team = combined[0][1]
            
            # new_team_scores = team_score_calcs(comp_format,team,database,print_table=False)
        
            #created table 
            tlas=['FX','PH','SR','VT','PB','HB','AA']
            team = combined[i][1]
            team_scores = team_score_calcs(comp_format,team,database,results=results_value,print_table=False)
            table = []
            
            for athlete in team:
                new_line = {}
                new_line['Athlete'] = athlete
                for tla in tlas:
                    #choose colour based on count 
                    #new_line.append(team_scores[athlete][tla][0])
                    #new_line.append(team_scores[athlete][tla][1])
                    # new_line.append(colored(team_scores[athlete][tla][0],colour_dict[team_scores[athlete][tla][1]]))
                    
                    # print('team_scores[athlete][tla]')
                    # print(team_scores[athlete][tla])
                    try:
                        new_line[tla]=team_scores[athlete][tla][0]
                    except:
                        new_line[tla]=team_scores[athlete][tla]
                    # new_line.append(colored(team_scores[athlete][tla][1],colour_dict[team_scores[athlete][tla][1]]))
                    
                    # header.append(tla)
                    # header.append("count")
                #We also add their AA scores
                # new_line.append(team_scores[athlete]['AA'])
                
                table.append(new_line)
            #summary line
            summary_line = {} #["Team"]
            summary_line['Athlete'] = 'Team Total'
            for tla in tlas:
                summary_line[tla] = np.round(team_scores['Team'][tla],3)
            table.append(summary_line)
            # print('team scores')
            # print(team_scores)
            # print('table')
            # print(table)
            table_data = table
            # table_data = team_score_dummy  # For now, using the same data for all tables
            # Truncate all numerical values to three decimal places
            for row in table_data:
                for key, value in row.items():
                    if isinstance(value, (int, float)):
                        row[key] = "{:.3f}".format(value)
            
            table = generate_table(table_data)
            tables.append(html.Div([html.H3(f'Team Scenario {i+1} using {results_value} results and {xx_value}-{yy_value}-{zz_value} competition format: {table_data[-1]["AA"]}'), table]))  # Add a heading for each table
            
            #rest clicks to get calculate button back
            # reset_n_clicks(n_clicks)

    return tables

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

#%% comment out when pusing to github
if __name__ == '__main__':
    app.run_server(debug=True)














