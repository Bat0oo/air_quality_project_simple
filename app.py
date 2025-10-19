import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime

# Load and prepare data
try:
    # Load all three CSV files from dataset folder
    beograd_data = pd.read_csv('dataset/beograd.csv')
    novi_sad_data = pd.read_csv('dataset/novi_sad.csv')
    nis_data = pd.read_csv('dataset/nis.csv')
    
    # Combine all data
    all_data = pd.concat([beograd_data, novi_sad_data, nis_data], ignore_index=True)
    
    # Drop rows with missing values
    all_data = all_data.dropna(subset=['value', 'datetimeLocal'])
    
    # Convert datetime to pandas datetime with UTC
    all_data['datetime'] = pd.to_datetime(all_data['datetimeLocal'], utc=True, errors='coerce')
    
    # Drop rows where datetime conversion failed
    all_data = all_data.dropna(subset=['datetime'])
    
    # Extract time features
    all_data['hour'] = all_data['datetime'].dt.hour
    all_data['day'] = all_data['datetime'].dt.day
    all_data['month'] = all_data['datetime'].dt.month
    all_data['day_of_week'] = all_data['datetime'].dt.dayofweek
    
    # Get unique cities and parameters
    cities = all_data['location_name'].unique()
    parameters = all_data['parameter'].unique()
    
    print(f"Loaded data from 3 cities: {cities}")
    print(f"Parameters: {parameters}")
    print(f"Total records: {len(all_data)}")
    
    # Train a model for each parameter
    models = {}
    
    for param in parameters:
        print(f"Training model for {param}...")
        param_data = all_data[all_data['parameter'] == param].copy()
        
        # Prepare features
        param_data = pd.get_dummies(param_data, columns=['location_name'])
        
        # Select feature columns
        feature_cols = ['hour', 'day', 'month', 'day_of_week'] + [col for col in param_data.columns if col.startswith('location_name_')]
        X = param_data[feature_cols]
        y = param_data['value']
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        models[param] = {
            'model': model,
            'feature_cols': feature_cols,
            'score': model.score(X_test, y_test)
        }
        print(f"{param} model trained. R² score: {models[param]['score']:.4f}")
    
except FileNotFoundError as e:
    print(f"Error: CSV file not found - {e}")
    exit()
except KeyError as e:
    print(f"Error: Missing column in CSV - {e}")
    exit()

# Create Flask app
app = Flask(__name__)

# Generate visualizations
def generate_city_comparison():
    """Bar chart comparing average values by city for each parameter"""
    city_averages = all_data.groupby(['location_name', 'parameter'])['value'].mean().reset_index()
    
    fig = px.bar(
        city_averages,
        x='location_name',
        y='value',
        color='parameter',
        barmode='group',
        title='Average Air Quality Parameters by City',
        labels={'location_name': 'City', 'value': 'Average Value (µg/m³)', 'parameter': 'Parameter'}
    )
    
    fig.update_layout(xaxis_title='City', yaxis_title='Average Value (µg/m³)')
    return pio.to_html(fig, full_html=False)

def generate_hourly_patterns():
    """Line chart showing hourly patterns for each parameter"""
    hourly_avg = all_data.groupby(['hour', 'parameter'])['value'].mean().reset_index()
    
    fig = px.line(
        hourly_avg,
        x='hour',
        y='value',
        color='parameter',
        title='Average Air Quality by Hour of Day',
        labels={'hour': 'Hour of Day', 'value': 'Average Value (µg/m³)', 'parameter': 'Parameter'}
    )
    
    fig.update_layout(xaxis_title='Hour of Day', yaxis_title='Average Value (µg/m³)')
    fig.update_xaxes(dtick=1)
    return pio.to_html(fig, full_html=False)

def generate_daily_patterns():
    """Line chart showing day of week patterns"""
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = all_data.groupby(['day_of_week', 'parameter'])['value'].mean().reset_index()
    daily_avg['day_name'] = daily_avg['day_of_week'].apply(lambda x: day_names[x])
    
    fig = px.line(
        daily_avg,
        x='day_name',
        y='value',
        color='parameter',
        title='Average Air Quality by Day of Week',
        labels={'day_name': 'Day of Week', 'value': 'Average Value (µg/m³)', 'parameter': 'Parameter'}
    )
    
    fig.update_layout(xaxis_title='Day of Week', yaxis_title='Average Value (µg/m³)')
    return pio.to_html(fig, full_html=False)

def generate_monthly_patterns():
    """Line chart showing monthly patterns"""
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_avg = all_data.groupby(['month', 'parameter'])['value'].mean().reset_index()
    monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: month_names[x-1])
    
    fig = px.line(
        monthly_avg,
        x='month_name',
        y='value',
        color='parameter',
        title='Average Air Quality by Month',
        labels={'month_name': 'Month', 'value': 'Average Value (µg/m³)', 'parameter': 'Parameter'}
    )
    
    fig.update_layout(xaxis_title='Month', yaxis_title='Average Value (µg/m³)')
    return pio.to_html(fig, full_html=False)

def generate_parameter_distribution():
    """Box plot showing distribution of each parameter"""
    fig = px.box(
        all_data,
        x='parameter',
        y='value',
        color='parameter',
        title='Distribution of Air Quality Parameters',
        labels={'parameter': 'Parameter', 'value': 'Value (µg/m³)'}
    )
    
    fig.update_layout(xaxis_title='Parameter', yaxis_title='Value (µg/m³)')
    return pio.to_html(fig, full_html=False)

# Generate all charts
print("Generating visualizations...")
city_chart = generate_city_comparison()
hourly_chart = generate_hourly_patterns()
daily_chart = generate_daily_patterns()
monthly_chart = generate_monthly_patterns()
distribution_chart = generate_parameter_distribution()
print("Visualizations ready!")

@app.route('/', methods=['GET', 'POST'])
def predict():
    results = None
    
    if request.method == 'POST':
        try:
            # Get input values
            city = request.form['city']
            hour = int(request.form['hour'])
            day = int(request.form['day'])
            month = int(request.form['month'])
            day_of_week = int(request.form['day_of_week'])
            
            results = {}
            
            # Predict for each parameter
            for param, model_info in models.items():
                # Create input dataframe
                input_data = pd.DataFrame([[hour, day, month, day_of_week]], 
                                         columns=['hour', 'day', 'month', 'day_of_week'])
                
                # Add city columns
                for col in model_info['feature_cols']:
                    if col.startswith('location_name_'):
                        input_data[col] = 0
                
                # Set the selected city to 1
                city_col = f'location_name_{city}'
                if city_col in input_data.columns:
                    input_data[city_col] = 1
                
                # Ensure all feature columns are present
                for col in model_info['feature_cols']:
                    if col not in input_data.columns:
                        input_data[col] = 0
                
                # Reorder columns to match training data
                input_data = input_data[model_info['feature_cols']]
                
                # Make prediction
                prediction = model_info['model'].predict(input_data)[0]
                results[param] = f'{prediction:.2f}'
                
        except Exception as e:
            results = {'Error': str(e)}
    
    return render_template(
        'index.html', 
        results=results, 
        cities=cities,
        city_chart=city_chart,
        hourly_chart=hourly_chart,
        daily_chart=daily_chart,
        monthly_chart=monthly_chart,
        distribution_chart=distribution_chart,
        model_info=models
    )

if __name__ == '__main__':
    app.run(debug=True)