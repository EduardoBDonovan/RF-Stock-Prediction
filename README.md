# Stock Market Prediction Using Machine Learning

This project utilizes historical stock market data from the S&P 500 (^GSPC), FTSE (^FTSE), and Dow Jones Industrial Average (^DJI) indices to predict market trends using a machine learning approach.

## Purpose
The purpose of this project is to demonstrate a basic stock market prediction model using a Random Forest Classifier. It employs historical stock data to train the model and predict whether the market will go up or down based on specified parameters.

## Dependencies
- Python 3.x
- Required Python libraries: `yfinance`, `pandas`, `scikit-learn`

## Usage
1. **Installation**
    - Ensure Python is installed.
    - Install required libraries: `yfinance`, `pandas`, `scikit-learn`.

2. **Code Overview**
    - The code is structured into sections:
        - **Data Retrieval and Cleaning:** Fetches historical data and prepares it for analysis.
        - **Model Training:** Trains a Random Forest Classifier on the prepared data.
        - **Prediction and Backtesting:** Predicts market trends and performs backtesting.

3. **Running the Code**
    - Ensure all dependencies are installed.
    - Run the code in a Python environment.
    - The output will display the precision score of the model and a directional prediction ("UP" or "Down").

4. **Customization**
    - Modify parameters such as the number of estimators in the Random Forest Classifier or the rolling averages used for predictions.

## Disclaimer
This project is for educational purposes only and does not guarantee accurate stock market predictions. Always perform due diligence and consider professional financial advice before making any investment decisions. This project was adapted from Dataquestio / project-walkthroughs / sp500.
