import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

## Retreiving Data
sp500 = yf.Ticker("^GSPC").history(period="max")
ftse = yf.Ticker("^FTSE").history(period="max")
djia = yf.Ticker("^DJI").history(period="max")

## Cleaning Up Data
del sp500["Dividends"]
del sp500["Stock Splits"]

del ftse["Dividends"]
del ftse["Stock Splits"]

del djia["Dividends"]
del djia["Stock Splits"]

## Creating Target and Tomorrow columns
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()

ftse["Tomorrow"] = ftse["Close"].shift(-1)
ftse["Target"] = (ftse["Tomorrow"] > ftse["Close"]).astype(int)
ftse = ftse.loc["1990-01-01":].copy()

djia["Tomorrow"] = djia["Close"].shift(-1)
djia["Target"] = (djia["Tomorrow"] > djia["Close"]).astype(int)
djia = djia.loc["1990-01-01":].copy()

## All rows except last 100 in train, last 100 in test
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    
    #Must be 60% sure in order for the prediction to be 1
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

## Take 10 years of data and train first model 2500 (start), 250 trading days a year (step)
## Test set is current year
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

## A variety of rolling averages

horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    
    ftse = ftse.reindex(sp500.index, method="ffill")
    
    sp_rolling_averages = sp500.rolling(horizon).mean()
    ftse_rolling_averages = ftse.rolling(horizon).mean()
    djia_rolling_averages = djia.rolling(horizon).mean()
    
    sp_ratio_column = f"SP_Close_Ratio_{horizon}"
    sp500[sp_ratio_column] = sp500["Close"] / sp_rolling_averages["Close"]
    
    ftse_ratio_column = f"FTSE_Close_Ratio_{horizon}"
    sp500[ftse_ratio_column] = ftse["Close"] / ftse_rolling_averages["Close"]
    
    djia_ratio_column = f"DJIA_Close_Ratio_{horizon}"
    sp500[djia_ratio_column] = djia["Close"] / djia_rolling_averages["Close"]
    
    sp_trend_column = f"SP_Trend_{horizon}"
    sp500[sp_trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    
    ftse_trend_column = f"FTSE_Trends_{horizon}"
    sp500[ftse_trend_column] = ftse.shift(1).rolling(horizon).sum()["Target"]
    
    djia_trend_column = f"DJIA_Trends_{horizon}"
    sp500[djia_trend_column] = djia.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors += [sp_ratio_column, ftse_ratio_column, djia_ratio_column, sp_trend_column, ftse_trend_column, djia_trend_column]
    
sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

predictions = backtest(sp500, model, new_predictors)

precision = (precision_score(predictions["Target"], predictions["Predictions"])) * 100

print(f"{precision}%")

if predictions["Predictions"].tail(1).iloc[-1] == 1:
    print("UP")
else:
    print("Down")