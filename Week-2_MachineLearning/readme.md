# IPL Win Prediction using Logistic Regression

This repository contains an example of a machine learning model (Logistic Regression) for predicting IPL match outcomes. It includes a Python notebook that details the process of data preparation and model training, as well as instructions on how to save the trained model.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

To set up the project, follow these steps:

1. Create a virtual environment.
2. Install the required packages in the environment:

   ```bash
   pip install -r requirements.txt
3. Download the dataset from [here](https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set) and place it in the `dataset` folder. (if you want to retrain the model)
4. Run the `eda.ipynb` file for training the model.
5. Run the `app.py` file to start the application.
6. output provides the link of localhost to access the application.

## Usage

The application is designed to predict the outcome of an IPL match based on the team's performance in previous seasons. The user can input the number of runs scored, and the number of runs to chase, overs remaining, wickets remaining and the application will return the predicted outcome.

