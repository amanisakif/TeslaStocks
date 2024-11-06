# TeslaStocks
# Tesla Stock Price Prediction

This project analyzes Tesla's historical stock data and implements a machine learning model to predict the stock's adjusted closing price. The project uses exploratory data analysis (EDA) to uncover trends and relationships within the data, and a machine learning model (Random Forest Regressor) to forecast stock prices.

## Project Overview

This project includes:
- **Data Cleaning and Preparation**: Loads, inspects, and preprocesses data to ensure accuracy.
- **Exploratory Data Analysis (EDA)**: Visualizes trends and correlations within Tesla's stock data.
- **Feature Engineering**: Creates lagged features to assist in predicting future values.
- **Machine Learning Model**: A Random Forest Regressor is used to predict the adjusted closing price.
- **Model Evaluation**: Uses Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) as evaluation metrics.

## Dataset

- **Source**: The dataset used is historical data for Tesla’s stock price.
- **Columns**:
  - `Date`: Trading date.
  - `Open`, `High`, `Low`, `Close`: Stock prices at open, highest, lowest, and close for each day.
  - `Adj Close`: Adjusted closing price.
  - `Volume`: Number of shares traded.

## Project Structure

- **Data Preparation**: Converts the `Date` column to datetime format, checks for missing values, and drops any unnecessary columns.
- **EDA**: 
  - Time series visualization of adjusted close prices.
  - Correlation heatmap for understanding relationships between variables.
- **Modeling**:
  - Splits data into training and testing sets.
  - Trains a Random Forest Regressor on historical stock data.
  - Evaluates performance using MAE, MSE, and RMSE.
- **Results**:
  - The model achieved an MAE of 2.25, MSE of 26.49, and RMSE of 5.15, which indicate reasonable accuracy depending on Tesla’s price range.

## Key Files

- `TSLA_stock_data.csv`: Historical stock data for Tesla (uploaded by the user).
- `analysis_and_prediction.ipynb`: Jupyter Notebook containing all code for data cleaning, EDA, model training, and evaluation.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/tesla-stock-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd tesla-stock-prediction
    ```
3. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run EDA**: Explore the dataset to visualize trends and understand key statistics.
2. **Train Model**: Use the prepared data to train the Random Forest Regressor.
3. **Evaluate Model**: Review the model's performance metrics (MAE, MSE, RMSE).
4. **Make Predictions**: Use the trained model to predict stock prices.

## Dependencies

- `pandas`: Data manipulation and analysis.
- `plotly`: Interactive visualizations.
- `matplotlib` and `seaborn`: Static visualizations.
- `sklearn`: Machine learning and evaluation metrics.
- `numpy`: Numeric operations.

Install dependencies with:
```bash
pip install pandas plotly matplotlib seaborn scikit-learn numpy
