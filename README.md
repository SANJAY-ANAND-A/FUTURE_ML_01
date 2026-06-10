# FUTURE_ML_01 - Sales & Demand Forecasting for Businesses

## Project Overview

This project was completed as part of the Future Interns Machine Learning Track - Task 1.

The objective of this project is to build a sales forecasting model using historical business sales data. Sales forecasting helps businesses plan inventory, manage cash flow, prepare staffing, and reduce losses caused by overstocking or stock shortages.

## Dataset

Dataset used: Superstore Sales Dataset

The dataset contains historical retail order records, including order dates, sales values, product categories, customer segments, and regions.

## Tools and Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code Notebook

## Key Steps

1. Loaded and explored the sales dataset
2. Cleaned and prepared date-based sales data
3. Aggregated order-level sales into monthly sales
4. Created time-based features such as year, month, quarter, and time index
5. Trained a Linear Regression forecasting model
6. Evaluated model performance using MAE, RMSE, and R2 Score
7. Forecasted sales for the next 6 months
8. Created business-friendly visualizations

## Model Performance

- Mean Absolute Error (MAE): 12211.04
- Root Mean Squared Error (RMSE): 17043.04
- R2 Score: 0.5627

## Business Insights

The model provides a monthly sales forecast that can help a business manager estimate future demand. These predictions can support:

- Inventory planning
- Cash flow management
- Staffing decisions
- Sales target setting
- Reducing overstocking and stock shortages

## Output Files

- `notebooks/sales_forecasting_task1.ipynb` - Complete notebook with code, model, visualizations, and explanation
- `outputs/future_sales_forecast.csv` - Forecasted sales for the next 6 months
- `images/actual_vs_predicted_sales.png` - Actual vs predicted sales chart
- `images/future_sales_forecast.png` - Future sales forecast chart

## Conclusion

This project demonstrates how machine learning can support business decision-making through sales and demand forecasting. The forecast is presented with clear visuals and simple interpretation so that non-technical stakeholders can understand and use the results.