import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64

# Initialize the Dash app with a modern theme
app = dash.Dash(__name__, external_stylesheets=[
    'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
])

# Load and preprocess the data
df = pd.read_csv("billboard_songs_with_views_likes_genre.csv")
df['date'] = pd.to_datetime(df['date'])
df['popularity_score'] = (df['views'] * 0.6 + df['likes'] * 0.4) / df['rank']
df['engagement_ratio'] = df['likes'] / df['views']

# Generate random duration data for demonstration (since it's not in the original dataset)
np.random.seed(42)
df['duration_minutes'] = np.random.uniform(2.5, 5.5, size=len(df))
df['release_month'] = df['date'].dt.month
df['release_day'] = df['date'].dt.day

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1([
                html.I(className="fas fa-music", style={"marginRight": "15px"}),
                "BeatsInsights"
            ], className='header-title'),
            html.P("An Intelligent Music Analytics Dashboard", className='header-description'),
        ], className="header-content"),
    ], className='header'),

    # Main content container
    html.Div([
        # Left column - Filters and KPIs
        html.Div([
            html.Div([
                html.H3([
                    html.I(className="fas fa-filter", style={"marginRight": "10px"}),
                    "Filters"
                ], className='section-title'),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=df['date'].min(),
                    end_date=df['date'].max(),
                    className='date-picker'
                ),
                html.Div([
                    html.H4([
                        html.I(className="fas fa-guitar", style={"marginRight": "8px"}),
                        "Genre Filter"
                    ]),
                    dcc.Dropdown(
                        id='genre-filter',
                        options=[{'label': x, 'value': x} for x in sorted(df['genre'].unique())],
                        value=[],
                        multi=True,
                        placeholder="Select Genres",
                        className='filter-dropdown'
                    ),
                ], className='filter-section'),
                
                html.Div([
                    html.H4([
                        html.I(className="fas fa-user", style={"marginRight": "8px"}),
                        "Artist Filter"
                    ]),
                    dcc.Dropdown(
                        id='artist-filter',
                        options=[{'label': x, 'value': x} for x in sorted(df['artist'].unique())],
                        value=[],
                        multi=True,
                        placeholder="Select Artists",
                        className='filter-dropdown'
                    ),
                ], className='filter-section'),
                
                html.Div([
                    html.H4([
                        html.I(className="fas fa-globe", style={"marginRight": "8px"}),
                        "Region Filter"
                    ]),
                    dcc.Dropdown(
                        id='region-filter',
                        options=[{'label': x, 'value': x} for x in sorted(df['region'].unique())],
                        value=[],
                        multi=True,
                        placeholder="Select Regions",
                        className='filter-dropdown'
                    ),
                ], className='filter-section'),
                
                html.Button([
                    html.I(className="fas fa-check", style={"marginRight": "8px"}),
                    'Apply Filters'
                ], id='apply-filters', n_clicks=0, className='filter-button'),
            ], className='filters-container'),

            # KPI Cards
            html.Div([
                html.Div([
                    html.I(className="fas fa-eye kpi-icon"),
                    html.H4("Total Views"),
                    html.H2(id='total-views', children="0", style={"fontSize": "24px", "whiteSpace": "normal", "wordBreak": "break-word"})
                ], className='kpi-card'),
                html.Div([
                    html.I(className="fas fa-heart kpi-icon"),
                    html.H4("Total Likes"),
                    html.H2(id='total-likes', children="0", style={"fontSize": "24px", "whiteSpace": "normal", "wordBreak": "break-word"})
                ], className='kpi-card'),
                html.Div([
                    html.I(className="fas fa-chart-pie kpi-icon"),
                    html.H4("Avg. Engagement"),
                    html.H2(id='avg-engagement', children="0%")
                ], className='kpi-card'),
                html.Div([
                    html.I(className="fas fa-guitar kpi-icon"),
                    html.H4("Top Genre"),
                    html.H2(id='top-genre', children="-")
                ], className='kpi-card'),
            ], className='kpi-container'),
        ], className='left-column'),

        # Right column - Charts
        html.Div([
            # Top charts row
            html.Div([
                # Genre Trends Chart
                html.Div([
                    html.H3([
                        html.I(className="fas fa-chart-line", style={"marginRight": "10px"}),
                        "Genre Popularity Trends"
                    ], className='chart-title'),
                    dcc.Graph(id='genre-trends-chart')
                ], className='chart-container half-width'),
                
                # Engagement vs Views Scatter Plot
                html.Div([
                    html.H3([
                        html.I(className="fas fa-bullseye", style={"marginRight": "10px"}),
                        "Engagement vs Views Analysis"
                    ], className='chart-title'),
                    dcc.Graph(id='engagement-scatter-plot')
                ], className='chart-container half-width'),
            ], className='top-charts'),

            # Second row
            html.Div([
                # Top Artists Chart
                html.Div([
                    html.H3([
                        html.I(className="fas fa-crown", style={"marginRight": "10px"}),
                        "Top Artists"
                    ], className='chart-title'),
                    dcc.Graph(id='top-artists-chart')
                ], className='chart-container half-width'),
                
                # Regional Analysis Chart
                html.Div([
                    html.H3([
                        html.I(className="fas fa-globe-americas", style={"marginRight": "10px"}),
                        "Regional Analysis"
                    ], className='chart-title'),
                    dcc.Graph(id='regional-analysis-chart')
                ], className='chart-container half-width'),
            ], className='middle-charts'),

            # Third row
            html.Div([
                # Song Comparison Chart
                html.Div([
                    html.H3([
                        html.I(className="fas fa-exchange-alt", style={"marginRight": "10px"}),
                        "Song Comparison"
                    ], className='chart-title'),
                    dcc.Dropdown(
                        id='song-comparison-selector',
                        options=[{'label': x, 'value': x} for x in sorted(df['song'].unique())],
                        value=[],
                        multi=True,
                        placeholder="Select Songs to Compare",
                        className='comparison-dropdown'
                    ),
                    dcc.Graph(id='song-comparison-chart')
                ], className='chart-container half-width'),
                
                # Artist Comparison Chart
                html.Div([
                    html.H3([
                        html.I(className="fas fa-users", style={"marginRight": "10px"}),
                        "Artist Comparison"
                    ], className='chart-title'),
                    dcc.Dropdown(
                        id='artist-comparison-selector',
                        options=[{'label': x, 'value': x} for x in sorted(df['artist'].unique())],
                        value=[],
                        multi=True,
                        placeholder="Select Artists to Compare",
                        className='comparison-dropdown'
                    ),
                    dcc.Graph(id='artist-comparison-chart')
                ], className='chart-container half-width'),
            ], className='comparison-charts'),

            # Fourth row
            html.Div([
                # Genre Distribution Pie Chart
                html.Div([
                    html.H3([
                        html.I(className="fas fa-chart-pie", style={"marginRight": "10px"}),
                        "Genre Distribution"
                    ], className='chart-title'),
                    dcc.Graph(id='genre-distribution-chart')
                ], className='chart-container half-width'),
                
                # Views Over Time
                html.Div([
                    html.H3([
                        html.I(className="fas fa-chart-line", style={"marginRight": "10px"}),
                        "Views Over Time"
                    ], className='chart-title'),
                    dcc.Graph(id='views-time-chart')
                ], className='chart-container half-width'),
            ], className='fourth-charts'),

            # Fifth row
            html.Div([
                # Likes Over Time
                html.Div([
                    html.H3([
                        html.I(className="fas fa-heart", style={"marginRight": "10px"}),
                        "Likes Over Time"
                    ], className='chart-title'),
                    dcc.Graph(id='likes-time-chart')
                ], className='chart-container half-width'),
                
                # Engagement Trends
                html.Div([
                    html.H3([
                        html.I(className="fas fa-chart-bar", style={"marginRight": "10px"}),
                        "Engagement Trends"
                    ], className='chart-title'),
                    dcc.Graph(id='engagement-trends-chart')
                ], className='chart-container half-width'),
            ], className='fifth-charts'),

            # Sixth row - New charts
            html.Div([
                # Song Duration Analysis
                html.Div([
                    html.H3([
                        html.I(className="fas fa-clock", style={"marginRight": "10px"}),
                        "Song Duration Analysis"
                    ], className='chart-title'),
                    dcc.Graph(id='duration-analysis-chart')
                ], className='chart-container half-width'),
                
                # Music Release Patterns
                html.Div([
                    html.H3([
                        html.I(className="fas fa-calendar-alt", style={"marginRight": "10px"}),
                        "Music Release Patterns"
                    ], className='chart-title'),
                    dcc.Graph(id='release-patterns-chart')
                ], className='chart-container half-width'),
            ], className='sixth-charts'),

            # Seventh row - New charts
            html.Div([
                # Artist Collaboration Network
                html.Div([
                    html.H3([
                        html.I(className="fas fa-project-diagram", style={"marginRight": "10px"}),
                        "Artist Collaboration Network"
                    ], className='chart-title'),
                    dcc.Graph(id='collaboration-network-chart')
                ], className='chart-container half-width'),
                
                # Top Songs Table
                html.Div([
                    html.H3([
                        html.I(className="fas fa-list", style={"marginRight": "10px"}),
                        "Top Songs"
                    ], className='chart-title'),
                    html.Div(id='music-names-table', className='table-container')
                ], className='chart-container half-width'),
            ], className='seventh-charts'),
        ], className='right-column'),
    ], className='main-content'),
], className='app-container')

# Callback for updating KPIs
@app.callback(
    [Output('total-views', 'children'),
     Output('total-likes', 'children'),
     Output('avg-engagement', 'children'),
     Output('top-genre', 'children')],
    [Input('apply-filters', 'n_clicks')],
    [State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('genre-filter', 'value'),
     State('artist-filter', 'value'),
     State('region-filter', 'value')]
)
def update_kpis(n_clicks, start_date, end_date, selected_genres, selected_artists, selected_regions):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & 
                                (filtered_df['date'] <= end_date)]
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
    
    if selected_artists:
        filtered_df = filtered_df[filtered_df['artist'].isin(selected_artists)]
        
    if selected_regions:
        filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]
    
    total_views = filtered_df['views'].sum()
    total_likes = filtered_df['likes'].sum()
    avg_engagement = (total_likes / total_views * 100) if total_views > 0 else 0
    
    top_genre = filtered_df.groupby('genre')['popularity_score'].mean().idxmax()
    
    return (f"{total_views:,.0f}",
            f"{total_likes:,.0f}",
            f"{avg_engagement:.1f}%",
            top_genre)

# Callback for Genre Trends Chart
@app.callback(
    Output('genre-trends-chart', 'figure'),
    [Input('apply-filters', 'n_clicks')],
    [State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('genre-filter', 'value'),
     State('artist-filter', 'value'),
     State('region-filter', 'value')]
)
def update_genre_trends(n_clicks, start_date, end_date, selected_genres, selected_artists, selected_regions):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & 
                                (filtered_df['date'] <= end_date)]
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
    
    if selected_artists:
        filtered_df = filtered_df[filtered_df['artist'].isin(selected_artists)]
        
    if selected_regions:
        filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]
    
    genre_trends = filtered_df.groupby(['date', 'genre'])['popularity_score'].mean().reset_index()
    
    fig = px.line(genre_trends, 
                  x='date', 
                  y='popularity_score',
                  color='genre',
                  title='Genre Popularity Over Time')
    
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='Popularity Score',
        showlegend=True,
        legend_title='Genre',
        hovermode='x unified'
    )
    
    return fig

# Callback for Top Artists Chart
@app.callback(
    Output('top-artists-chart', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('genre-filter', 'value')]
)
def update_top_artists(start_date, end_date, selected_genres):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & 
                                (filtered_df['date'] <= end_date)]
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
    
    top_artists = (filtered_df.groupby('artist')['popularity_score']
                  .mean()
                  .sort_values(ascending=True)
                  .tail(10))
    
    fig = go.Figure(go.Bar(
        x=top_artists.values,
        y=top_artists.index,
        orientation='h'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Average Popularity Score',
        yaxis_title='Artist',
        showlegend=False,
        height=400
    )
    
    return fig

# Callback for Regional Analysis Chart
@app.callback(
    Output('regional-analysis-chart', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('genre-filter', 'value')]
)
def update_regional_analysis(start_date, end_date, selected_genres):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & 
                                (filtered_df['date'] <= end_date)]
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
    
    regional_data = filtered_df.groupby('region')['popularity_score'].mean().sort_values(ascending=True)
    
    fig = go.Figure(go.Bar(
        x=regional_data.values,
        y=regional_data.index,
        orientation='h'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Average Popularity Score',
        yaxis_title='Region',
        showlegend=False,
        height=400
    )
    
    return fig

# New callback for Engagement Scatter Plot
@app.callback(
    Output('engagement-scatter-plot', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('genre-filter', 'value')]
)
def update_engagement_scatter(start_date, end_date, selected_genres):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & 
                                (filtered_df['date'] <= end_date)]
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
    
    fig = px.scatter(filtered_df,
                    x='views',
                    y='likes',
                    color='genre',
                    size='popularity_score',
                    hover_data=['song', 'artist'],
                    title='Engagement Analysis')
    
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Views',
        yaxis_title='Likes',
        showlegend=True,
        legend_title='Genre'
    )
    
    return fig

# New callback for Music Names Table
@app.callback(
    Output('music-names-table', 'children'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('genre-filter', 'value')]
)
def update_music_table(start_date, end_date, selected_genres):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & 
                                (filtered_df['date'] <= end_date)]
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
    
    # Get top 10 songs by popularity score
    top_songs = filtered_df.nlargest(10, 'popularity_score')[['song', 'artist', 'genre', 'rank', 'views', 'likes']]
    
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in ['Song', 'Artist', 'Genre', 'Rank', 'Views', 'Likes']])] +
        # Body
        [html.Tr([
            html.Td(row['song']),
            html.Td(row['artist']),
            html.Td(row['genre']),
            html.Td(row['rank']),
            html.Td(f"{row['views']:,.0f}"),
            html.Td(f"{row['likes']:,.0f}")
        ]) for _, row in top_songs.iterrows()],
        className='music-table'
    )

# Song Comparison Chart Callback
@app.callback(
    Output('song-comparison-chart', 'figure'),
    [Input('song-comparison-selector', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_song_comparison(selected_songs, start_date, end_date):
    if not selected_songs:
        return go.Figure()
        
    filtered_df = df[df['song'].isin(selected_songs)]
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & 
                                (filtered_df['date'] <= end_date)]
    
    fig = go.Figure()
    
    for song in selected_songs:
        song_data = filtered_df[filtered_df['song'] == song]
        fig.add_trace(go.Scatter(
            x=song_data['date'],
            y=song_data['views'],
            name=song,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        template='plotly_dark',
        title='Song Views Comparison Over Time',
        xaxis_title='Date',
        yaxis_title='Views',
        showlegend=True,
        legend_title='Songs'
    )
    
    return fig

# Artist Comparison Chart Callback
@app.callback(
    Output('artist-comparison-chart', 'figure'),
    [Input('artist-comparison-selector', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_artist_comparison(selected_artists, start_date, end_date):
    if not selected_artists:
        return go.Figure()
        
    filtered_df = df[df['artist'].isin(selected_artists)]
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & 
                                (filtered_df['date'] <= end_date)]
    
    artist_stats = filtered_df.groupby('artist').agg({
        'views': 'sum',
        'likes': 'sum',
        'popularity_score': 'mean'
    }).reset_index()
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Total Views', 'Total Likes', 'Avg Popularity'))
    
    fig.add_trace(
        go.Bar(x=artist_stats['artist'], y=artist_stats['views'], name='Views'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=artist_stats['artist'], y=artist_stats['likes'], name='Likes'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=artist_stats['artist'], y=artist_stats['popularity_score'], name='Popularity'),
        row=1, col=3
    )
    
    fig.update_layout(
        template='plotly_dark',
        showlegend=False,
        height=400
    )
    
    return fig

# Genre Distribution Chart Callback
@app.callback(
    Output('genre-distribution-chart', 'figure'),
    [Input('apply-filters', 'n_clicks')],
    [State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('artist-filter', 'value'),
     State('region-filter', 'value')]
)
def update_genre_distribution(n_clicks, start_date, end_date, selected_artists, selected_regions):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & 
                                (filtered_df['date'] <= end_date)]
    
    if selected_artists:
        filtered_df = filtered_df[filtered_df['artist'].isin(selected_artists)]
        
    if selected_regions:
        filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]
    
    genre_dist = filtered_df.groupby('genre')['views'].sum().reset_index()
    
    fig = px.pie(genre_dist, 
                 values='views', 
                 names='genre',
                 title='Genre Distribution by Views')
    
    fig.update_layout(template='plotly_dark')
    
    return fig

# Views Over Time Chart Callback
@app.callback(
    Output('views-time-chart', 'figure'),
    [Input('apply-filters', 'n_clicks')],
    [State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('genre-filter', 'value'),
     State('artist-filter', 'value'),
     State('region-filter', 'value')]
)
def update_views_time(n_clicks, start_date, end_date, selected_genres, selected_artists, selected_regions):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & 
                                (filtered_df['date'] <= end_date)]
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
    
    if selected_artists:
        filtered_df = filtered_df[filtered_df['artist'].isin(selected_artists)]
        
    if selected_regions:
        filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]
    
    views_time = filtered_df.groupby('date')['views'].sum().reset_index()
    
    # Create figure using go.Figure instead of px.line
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=views_time['date'],
        y=views_time['views'],
        mode='lines',
        name='Views'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        title='Total Views Over Time',
        xaxis_title='Date',
        yaxis_title='Total Views',
        showlegend=True
    )
    
    return fig

# Likes Over Time Chart Callback
@app.callback(
    Output('likes-time-chart', 'figure'),
    [Input('apply-filters', 'n_clicks')],
    [State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('genre-filter', 'value'),
     State('artist-filter', 'value'),
     State('region-filter', 'value')]
)
def update_likes_time(n_clicks, start_date, end_date, selected_genres, selected_artists, selected_regions):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & 
                                (filtered_df['date'] <= end_date)]
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
    
    if selected_artists:
        filtered_df = filtered_df[filtered_df['artist'].isin(selected_artists)]
        
    if selected_regions:
        filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]
    
    likes_time = filtered_df.groupby('date')['likes'].sum().reset_index()
    
    fig = px.line(likes_time, 
                  x='date', 
                  y='likes',
                  title='Total Likes Over Time')
    
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='Total Likes'
    )
    
    return fig

# Engagement Trends Chart Callback
@app.callback(
    Output('engagement-trends-chart', 'figure'),
    [Input('apply-filters', 'n_clicks')],
    [State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('genre-filter', 'value'),
     State('artist-filter', 'value'),
     State('region-filter', 'value')]
)
def update_engagement_trends(n_clicks, start_date, end_date, selected_genres, selected_artists, selected_regions):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & 
                                (filtered_df['date'] <= end_date)]
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
    
    if selected_artists:
        filtered_df = filtered_df[filtered_df['artist'].isin(selected_artists)]
        
    if selected_regions:
        filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]
    
    engagement_trends = filtered_df.groupby('date')['engagement_ratio'].mean().reset_index()
    
    fig = px.line(engagement_trends, 
                  x='date', 
                  y='engagement_ratio',
                  title='Average Engagement Ratio Over Time')
    
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='Engagement Ratio (Likes/Views)'
    )
    
    return fig

# Song Duration Analysis Chart Callback
@app.callback(
    Output('duration-analysis-chart', 'figure'),
    [Input('apply-filters', 'n_clicks')],
    [State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('genre-filter', 'value'),
     State('artist-filter', 'value'),
     State('region-filter', 'value')]
)
def update_duration_analysis(n_clicks, start_date, end_date, selected_genres, selected_artists, selected_regions):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & 
                                (filtered_df['date'] <= end_date)]
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
    
    if selected_artists:
        filtered_df = filtered_df[filtered_df['artist'].isin(selected_artists)]
        
    if selected_regions:
        filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]
    
    # Create a histogram of song durations
    fig = px.histogram(filtered_df, 
                      x='duration_minutes',
                      color='genre',
                      title='Song Duration Distribution by Genre',
                      labels={'duration_minutes': 'Duration (minutes)'},
                      nbins=20)
    
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Duration (minutes)',
        yaxis_title='Count',
        showlegend=True,
        legend_title='Genre'
    )
    
    return fig

# Music Release Patterns Chart Callback
@app.callback(
    Output('release-patterns-chart', 'figure'),
    [Input('apply-filters', 'n_clicks')],
    [State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('genre-filter', 'value'),
     State('artist-filter', 'value'),
     State('region-filter', 'value')]
)
def update_release_patterns(n_clicks, start_date, end_date, selected_genres, selected_artists, selected_regions):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & 
                                (filtered_df['date'] <= end_date)]
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
    
    if selected_artists:
        filtered_df = filtered_df[filtered_df['artist'].isin(selected_artists)]
        
    if selected_regions:
        filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]
    
    # Create a heatmap of release patterns by month and day
    release_patterns = filtered_df.groupby(['release_month', 'release_day']).size().reset_index(name='count')
    
    fig = px.density_heatmap(release_patterns, 
                            x='release_day', 
                            y='release_month',
                            z='count',
                            title='Music Release Patterns by Month and Day',
                            labels={'release_day': 'Day of Month', 'release_month': 'Month'})
    
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Day of Month',
        yaxis_title='Month'
    )
    
    return fig

# Artist Collaboration Network Chart Callback
@app.callback(
    Output('collaboration-network-chart', 'figure'),
    [Input('apply-filters', 'n_clicks')],
    [State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('genre-filter', 'value'),
     State('artist-filter', 'value'),
     State('region-filter', 'value')]
)
def update_collaboration_network(n_clicks, start_date, end_date, selected_genres, selected_artists, selected_regions):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & 
                                (filtered_df['date'] <= end_date)]
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
    
    if selected_artists:
        filtered_df = filtered_df[filtered_df['artist'].isin(selected_artists)]
        
    if selected_regions:
        filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]
    
    # For demonstration, create a simple network of top artists
    top_artists = filtered_df.groupby('artist')['views'].sum().nlargest(10).index.tolist()
    
    # Create a simple network visualization
    fig = go.Figure()
    
    # Add nodes (artists)
    for i, artist in enumerate(top_artists):
        fig.add_trace(go.Scatter(
            x=[np.cos(2 * np.pi * i / len(top_artists))],
            y=[np.sin(2 * np.pi * i / len(top_artists))],
            mode='markers+text',
            name=artist,
            text=[artist],
            textposition='top center',
            marker=dict(size=20, color='#3498db'),
            showlegend=False
        ))
    
    # Add edges (connections between artists)
    for i in range(len(top_artists)):
        for j in range(i+1, len(top_artists)):
            fig.add_trace(go.Scatter(
                x=[np.cos(2 * np.pi * i / len(top_artists)), np.cos(2 * np.pi * j / len(top_artists))],
                y=[np.sin(2 * np.pi * i / len(top_artists)), np.sin(2 * np.pi * j / len(top_artists))],
                mode='lines',
                line=dict(color='rgba(52, 152, 219, 0.5)', width=1),
                showlegend=False
            ))
    
    fig.update_layout(
        template='plotly_dark',
        title='Artist Collaboration Network',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=400
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True) 