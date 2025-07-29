# House Price Prediction

This project predicts house prices using property features such as size, bedrooms, and location. It demonstrates a complete machine learning workflow, including data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation using both Linear Regression and Gradient Boosting Regressor.

## Objective
Predict house prices based on various property features using regression models.

## Dataset
- **Source:** [Kaggle - Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/data)
- **Features:**
  - `price` (target)
  - `area`, `bedrooms`, `bathrooms`, `stories`, `parking`
  - `mainroad`, `guestroom`, `basement`, `airconditioning`, `prefarea`, `furnishingstatus` (categorical)

## Workflow
1. **Data Loading & Exploration:**
   - Loads the dataset and explores its structure, missing values, and unique values.
2. **Univariate Analysis:**
   - Visualizes distributions of numerical and categorical features.
3. **Feature Engineering:**
   - Drops low-variance features (e.g., `hotwaterheating`).
   - Encodes categorical variables (binary and one-hot encoding).
   - Handles outliers by capping extreme values.
   - Standardizes numerical features.
4. **Model Training:**
   - Splits data into training and testing sets.
   - Trains a Linear Regression model and a Gradient Boosting Regressor.
5. **Evaluation:**
   - Evaluates models using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
   - Visualizes actual vs. predicted prices.
   - Compares model performance.

## Results
- **Linear Regression:**
  - MAE: ~842,476
  - RMSE: ~1,076,085
- **Gradient Boosting Regressor:**
  - MAE: ~830,134
  - RMSE: ~1,068,091
- The Gradient Boosting Regressor performed marginally better.

## Requirements
- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/data) and place `Housing.csv` in the appropriate directory (update the path in the notebook if needed).
2. Open `House_Prices_Prediction.ipynb` in Jupyter Notebook.
3. Run the notebook cells sequentially to reproduce the analysis and results.

## Project Structure
```
/House-Price-Prediction
├── House_Prices_Prediction.ipynb
├── README.md
└── Housing.csv
```

## License
This project is for educational purposes. 