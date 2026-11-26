
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

# 1. Load monthly historical revenue
df = pd.read_csv("data/monthly_revenue.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# 2. Prepare time series
revenue = df.set_index("Date")["Revenue"]
revenue = revenue.asfreq("MS")

# 3. Train ARIMA using all 12 months
model = ARIMA(
    revenue,
    order=(1, 1, 0)
)

fitted_model = model.fit()

# 4. Forecast next three months
forecast_result = fitted_model.get_forecast(steps=3)

forecast_mean = forecast_result.predicted_mean
confidence_intervals = forecast_result.conf_int()

# 5. Create forecast report
forecast_df = pd.DataFrame({
    "Date": forecast_mean.index,
    "Predicted_Revenue": forecast_mean.values,
    "Lower_95": confidence_intervals.iloc[:, 0].values,
    "Upper_95": confidence_intervals.iloc[:, 1].values
})

print("\nNEXT THREE MONTHS REVENUE FORECAST")
print(forecast_df.to_string(index=False))

# 6. Save forecast
forecast_df.to_csv(
    "data/future_forecast.csv",
    index=False
)

# 7. Visualize historical and forecast revenue
plt.figure(figsize=(12, 6))

plt.plot(
    revenue.index,
    revenue.values,
    marker="o",
    label="Historical Revenue"
)

plt.plot(
    forecast_mean.index,
    forecast_mean.values,
    marker="o",
    linestyle="--",
    label="ARIMA Forecast"
)

plt.fill_between(
    forecast_mean.index,
    confidence_intervals.iloc[:, 0],
    confidence_intervals.iloc[:, 1],
    alpha=0.2,
    label="95% Prediction Interval"
)

plt.title("Sales Revenue Forecast - Jan to Mar 2026")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("future_revenue_forecast.png")
plt.show()

print("\nFuture forecasting completed!")
print("Forecast saved to data/future_forecast.csv")
