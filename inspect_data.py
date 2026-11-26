
import pandas as pd

# Load historical sales data
df = pd.read_csv("data/sales_data.csv")

print("\nFIRST FIVE RECORDS")
print(df.head())

print("\nDATASET DIMENSIONS")
print(df.shape)

print("\nCOLUMN INFORMATION")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDATE RANGE")
print(df["Order_Date"].min())
print(df["Order_Date"].max())

print("\nNUMERICAL SUMMARY")
print(df.describe())
