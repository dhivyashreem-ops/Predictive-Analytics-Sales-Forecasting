
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load historical sales data
df = pd.read_csv("data/sales_data.csv")

# 2. Convert Order_Date into datetime format
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

# 3. Validate important columns
df["Revenue"] = pd.to_numeric(
    df["Revenue"],
    errors="coerce"
)

df = df.dropna(subset=["Order_Date", "Revenue"])

# 4. Sort transactions chronologically
df = df.sort_values("Order_Date")

# 5. Aggregate daily transactions into monthly revenue
monthly_revenue = (
    df.set_index("Order_Date")["Revenue"]
    .resample("MS")
    .sum()
    .reset_index()
)

monthly_revenue.columns = ["Date", "Revenue"]

# 6. Add time-based features
monthly_revenue["Year"] = monthly_revenue["Date"].dt.year
monthly_revenue["Month"] = monthly_revenue["Date"].dt.month
monthly_revenue["Month_Name"] = (
    monthly_revenue["Date"].dt.strftime("%B")
)

# 7. Display monthly revenue
print("\nMONTHLY REVENUE DATA")
print(monthly_revenue.to_string(index=False))

# 8. Save processed dataset
monthly_revenue.to_csv(
    "data/monthly_revenue.csv",
    index=False
)

# 9. Display summary
print("\nMONTHLY DATASET SHAPE:", monthly_revenue.shape)
print("TOTAL ANNUAL REVENUE:", round(monthly_revenue["Revenue"].sum(), 2))
print("AVERAGE MONTHLY REVENUE:", round(monthly_revenue["Revenue"].mean(), 2))

# 10. Visualize monthly revenue
plt.figure(figsize=(12, 5))

plt.plot(
    monthly_revenue["Date"],
    monthly_revenue["Revenue"],
    marker="o"
)

plt.title("Monthly Sales Revenue - 2025")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("monthly_revenue_trend.png")
plt.show()

print("\nMonthly preprocessing completed successfully!")
