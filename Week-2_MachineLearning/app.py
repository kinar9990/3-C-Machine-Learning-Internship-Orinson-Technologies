from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

app = Flask(__name__)

# Load the pre-trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

@app.route('/')
def index():
    return render_template('index.html', teams=sorted(teams), cities=sorted(cities))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    batting_team = data['batting_team']
    bowling_team = data['bowling_team']
    selected_city = data['city']
    target = data['target']
    score = data['score']
    overs = data['overs']
    wickets = data['wickets']

    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = pipe.predict_proba(input_df)
    win = result[0][1]
    loss = result[0][0]

    return jsonify({
        'win_probability': round(win * 100, 2),
        'loss_probability': round(loss * 100, 2)
    })

@app.route('/visualizations')
def visualizations():
    
    return render_template('visualization.html')

if __name__ == '__main__':
    app.run(debug=True)
