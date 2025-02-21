from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

app = Flask(__name__)

data_folder = r"D:\Projects\Real Project\Portfolio Management\data"

stock_data = {}

files = os.listdir(data_folder)

for file in files:
    if file.endswith('.csv'):
        file_path = os.path.join(data_folder, file)
        stock_name = file.split('.')[0]  # Assuming the stock name is part of the filename
        try:
            df = pd.read_csv(file_path, index_col="datetime", parse_dates=["datetime"])
            stock_data[stock_name] = df["close"]
        except Exception as e:
            print(f"Error reading {file}: {e}")

data = pd.concat(stock_data.values(), axis=1)
data.columns = stock_data.keys()

def calculate_portfolio_with_xgboost(stock_symbol, monthly_investment, years=5):
    prices = data[stock_symbol]
    prices = prices.resample('ME').mean()  # Resample to monthly data
    
    prices = prices.dropna()  # Drop rows with NaN values
    prices = prices[~prices.isin([np.inf, -np.inf])]  # Remove infinite values

    features = []
    target = []
    
    for i in range(1, len(prices)-1):
        features.append([prices.iloc[i-1], prices.iloc[i]])  # Using previous month's and current month's prices as features
        target.append(prices.iloc[i+1] / prices.iloc[i] - 1)  # Percentage return (change in price)
    
    X = np.array(features)
    y = np.array(target)
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Target contains NaN or Inf values.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(objective='reg:squarederror', early_stopping_rounds=10)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    predicted_returns = model.predict(X_test)
    print(mean_squared_error(y_test,predicted_returns))
    
    predicted_returns = np.nan_to_num(predicted_returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    cumulative_returns = np.cumsum(predicted_returns)
    
    total_investment = monthly_investment * years * 12
    final_value = total_investment * (1 + cumulative_returns[-1])  # Total return added to total investment
    
    return final_value, total_investment, cumulative_returns

def generate_chart(cumulative_returns):
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns)
    plt.title("Portfolio Cumulative Returns Over Time")
    plt.xlabel("Months")
    plt.ylabel("Cumulative Returns")
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return chart

@app.route('/', methods=['GET', 'POST'])
def home():
    final_value = None
    total_investment = None
    chart = None
    error = None

    if request.method == 'POST':
        try:
            stock_symbol = request.form['stock_symbol']
            monthly_investment = float(request.form['investment_amount'])
            final_value, total_investment, cumulative_returns = calculate_portfolio_with_xgboost(stock_symbol, monthly_investment)
            chart = generate_chart(cumulative_returns)
        except Exception as e:
            error = str(e)

    return render_template('index.html', 
                           final_value=final_value, 
                           total_investment=total_investment, 
                           chart=chart, 
                           error=error, 
                           stocks=data.columns)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)


