# Stock Price Prediction

Predict the next day's closing price for a selected stock (e.g., Apple) using historical data and machine learning models.

## Project Overview
This project analyzes historical stock data using the `yfinance` library and predicts the next day's closing price for a chosen stock (default: Apple/AAPL) using a Linear Regression model. The workflow includes data fetching, preparation, model training, evaluation, and visualization of results.

## Features
- Fetches historical stock data for a specified ticker and date range
- Prepares data for machine learning by creating features and target variables
- Splits data into training and testing sets
- Trains a Linear Regression model (optionally extendable to Random Forest)
- Evaluates model performance using Mean Squared Error (MSE) and R-squared metrics
- Visualizes actual vs. predicted closing prices

## Setup Instructions
1. **Clone the repository** and navigate to the project directory.
2. **Install dependencies** (preferably in a virtual environment):
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn yfinance
   ```
3. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook Stock_Price_Prediction.ipynb
   ```

## Usage
- Run each cell in the notebook sequentially.
- The default configuration fetches Apple (AAPL) stock data from 2020-01-01 to 2023-12-31.
- The notebook will output model evaluation metrics and a scatter plot comparing actual vs. predicted prices.

## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- yfinance

## Results
- The Linear Regression model achieved a Mean Squared Error (MSE) of approximately 7.26 and an R-squared value of approximately 0.993 on the test data.
- Visualization shows a strong correlation between actual and predicted prices.

## Next Steps / Improvements
- Explore additional features (e.g., technical indicators, news sentiment)
- Try other regression models (e.g., Random Forest)
- Extend to other stocks or longer-term predictions
- Automate hyperparameter tuning and model selection

---

*This project is for educational purposes and should not be used for actual trading decisions.* 