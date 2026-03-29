# Equity Portfolio Optimization — BSE Small-Cap Stock Selection

A college data science project that builds a stock selection system for BSE small-cap equities using machine learning.

## Objective

Predict stock prices at T2 using T1 financial fundamentals, then rank and select the top 50 stocks using a composite scoring model, and construct an equal-weight portfolio.

## Pipeline

```
Multi-source CSV data (T1 & T2)
        ↓
Data Merging & Preprocessing
        ↓
Feature Selection (Mutual Information — top 75th percentile)
        ↓
Model Training (XGBoost & Gradient Boosting)
        ↓
Stock Filtering & Risk Classification
        ↓
Composite Scoring → Top 50 Stocks
        ↓
Portfolio Construction (₹10L equal-weight)
```

## Models

| Model | Description |
|-------|-------------|
| XGBoost Regressor | Tuned with L1/L2 regularization, shallow trees (max_depth=4) |
| Gradient Boosting Regressor | Final model used for predictions (sklearn, squared_error loss) |

## Composite Scoring

```
Final Score = 0.4 × Predicted Return + 0.3 × ROE + 0.2 × ROA + 0.1 × P/E
```

All metrics are min-max normalized before scoring.

## Risk Classification

| Risk Level | Criteria |
|------------|----------|
| Low Risk | Debt/Equity < 0.5, P/E between 10–25, ROE > 15% |
| Medium Risk | Debt/Equity 0.5–1.5, P/E between 25–50, ROE 10–15% |
| High Risk | Everything else |

## Results

| Metric | Value |
|--------|-------|
| Total Capital | ₹10,00,000 |
| Stocks Selected | 50 |
| Industries Covered | 19 |
| Average Predicted Return | ~81% |
| Portfolio Predicted Value | ~₹18.11L |
| Model Used (Final) | Gradient Boosting Regressor |

## Dataset

Multi-source CSVs covering BSE small-cap stocks across two time periods (T1 = training, T2 = evaluation):
- Annual & Quarterly P&L statements
- Balance sheet
- Cash flow statements
- Financial ratios
- Stock prices

> Data was loaded from Google Drive during development. Replace paths in the notebook to reproduce locally.

## Files

| File | Description |
|------|-------------|
| `equity_portfolio.ipynb` | Main notebook — full pipeline |
| `equity_dashboard.html` | Interactive dashboard of results |

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
scipy
```
