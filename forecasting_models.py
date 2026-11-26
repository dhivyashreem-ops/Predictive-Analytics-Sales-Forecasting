
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.arima.model import ARIMA

# 1. Load monthly revenue
df = pd.read_csv("data/monthly_revenue.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

y = df["Revenue"].values
X = np.arange(len(df)).reshape(-1, 1)

# 2. Chronological train-test split
train_size = 9

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

results = []
predictions = {}

# 3. Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)
predictions["Linear Regression"] = linear_pred

# 4. Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

predictions["Random Forest"] = rf_pred

# 5. ARIMA
arima_model = ARIMA(
    y_train,
    order=(1, 1, 0)
).fit()

arima_pred = arima_model.forecast(steps=len(y_test))
predictions["ARIMA"] = np.asarray(arima_pred)

# 6. Naive baseline
naive_pred = np.repeat(y_train[-1], len(y_test))
predictions["Naive Baseline"] = naive_pred

# 7. Evaluate all models
for name, pred in predictions.items():

    mae = mean_absolute_error(y_test, pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, pred)
    )

    mape = np.mean(
        np.abs((y_test - pred) / y_test)
    ) * 100

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_Percent": mape
    })

# 8. Display results
metrics = pd.DataFrame(results)
metrics = metrics.sort_values("MAE")

print("\nMODEL PERFORMANCE COMPARISON")
print(metrics.to_string(index=False))

metrics.to_csv(
    "data/model_comparison.csv",
    index=False
)

# 9. Save actual vs predicted results
test_results = pd.DataFrame({
    "Date": df["Date"].iloc[train_size:].values,
    "Actual_Revenue": y_test
})

for name, pred in predictions.items():
    test_results[name] = pred

test_results.to_csv(
    "data/test_predictions.csv",
    index=False
)

# 10. Plot actual versus predicted revenue
plt.figure(figsize=(12, 6))

plt.plot(
    df["Date"],
    y,
    marker="o",
    label="Actual Revenue"
)

for name, pred in predictions.items():
    plt.plot(
        df["Date"].iloc[train_size:],
        pred,
        marker="x",
        linestyle="--",
        label=name
    )

plt.axvline(
    df["Date"].iloc[train_size],
    linestyle=":",
    label="Test Period Starts"
)

plt.title("Actual vs Predicted Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("forecast_comparison.png")
plt.show()

# 11. Select the lowest-MAE model for demonstration
best_model_name = metrics.iloc[0]["Model"]

print("\nBest model on the three-month test period:")
print(best_model_name)

print("\nForecast evaluation completed!")
