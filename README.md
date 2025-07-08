# Music Analytics Dashboard 🎵

An interactive dashboard built with Dash and Plotly for analyzing Billboard songs data, including views, likes, genres, and regional trends.

## Features

- 📊 Real-time data filtering and visualization
- 📈 Genre popularity trends over time
- 🎨 Top artists performance analysis
- 🌍 Regional music analysis
- 📱 Responsive design for all screen sizes
- 🎯 Key performance indicators (KPIs)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd music_dashboard
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Ensure your data file `billboard_songs_with_views_likes_genre.csv` is in the project directory.

## Usage

1. Start the dashboard:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:8050/
```

## Dashboard Components

### Filters
- Date Range Selector: Filter data by specific time periods
- Genre Filter: Select one or multiple genres to analyze

### KPI Cards
- Total Views
- Total Likes
- Average Engagement Rate

### Visualizations
- Genre Popularity Trends Chart
- Top Artists Performance Chart
- Regional Analysis Chart

## Data Requirements

The dashboard expects a CSV file named `billboard_songs_with_views_likes_genre.csv` with the following columns:
- date
- artist
- song
- genre
- region
- views
- likes
- rank

## Technologies Used

- Python
- Dash
- Plotly
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Statsmodels

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 