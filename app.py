"""Dash App Layout With Figure and Slider

Source: https://dash.plotly.com/basic-callbacks#dash-app-layout-with-figure-and-slider
"""

# Import packages
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# Read external data into pandas DataFrame
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')

# Initialize the app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("A Minimal Plotly Dash application"),
    dcc.Graph(id='graph-with-slider'),
    dcc.Slider(
        df['year'].min(),
        df['year'].max(),
        step=None,
        value=df['year'].min(),
        marks={str(year): str(year) for year in df['year'].unique()},
        id='year-slider'
    ),
    html.A("Github", href="https://github.com/peterhaasme/deploy-plotly-dash")
])

# App callbacks
@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('year-slider', 'value'))
def update_figure(selected_year):
    filtered_df = df[df.year == selected_year]

    fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",
                     size="pop", color="continent", hover_name="country",
                     log_x=True, size_max=55)

    fig.update_layout(transition_duration=500)

    return fig

#need to run code below for running locally but remove when 
#running on the web
if __name__ == '__main__':
    app.run_server(debug=True)
    
    
