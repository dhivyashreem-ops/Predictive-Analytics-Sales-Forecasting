from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="Predictive Sales Analytics", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp{background:#F5F7FA}
.block-container{padding-top:1.7rem;padding-bottom:2rem;max-width:1450px}
h1{color:#17365D;font-weight:750}
h2,h3{color:#24476B;font-weight:650}
[data-testid="stMetric"]{background:#fff;border:1px solid #E1E8F0;padding:18px;border-radius:12px;box-shadow:0 2px 8px rgba(30,60,90,.06)}
[data-testid="stMetricLabel"]{color:#526579}
[data-testid="stMetricValue"]{color:#17365D;font-weight:700}
section[data-testid="stSidebar"]{background:#EDF2F8}
hr{border-color:#DCE3EB}
</style>
""", unsafe_allow_html=True)

def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        st.error(f"Required file not found: {path}")
        st.info("Run the data preparation, model comparison and forecasting scripts first.")
        st.stop()
    return pd.read_csv(path)

def find_column(df, candidates):
    normalized = {"".join(ch.lower() for ch in str(c) if ch.isalnum()): c for c in df.columns}
    for candidate in candidates:
        key = "".join(ch.lower() for ch in candidate if ch.isalnum())
        if key in normalized:
            return normalized[key]
    return None

def money_lakh(value):
    return f"₹{value / 100000:,.2f} L"

monthly_df = load_csv("monthly_revenue.csv")
forecast_df = load_csv("future_forecast.csv")
model_df = load_csv("model_comparison.csv")
predictions_df = load_csv("test_predictions.csv")

monthly_df["Date"] = pd.to_datetime(monthly_df["Date"], errors="coerce")
monthly_df["Revenue"] = pd.to_numeric(monthly_df["Revenue"], errors="coerce")
monthly_df = monthly_df.dropna(subset=["Date", "Revenue"]).sort_values("Date")
monthly_df["Month_Name"] = monthly_df["Date"].dt.strftime("%b %Y")
monthly_df["Revenue_Lakh"] = monthly_df["Revenue"] / 100000

forecast_df["Date"] = pd.to_datetime(forecast_df["Date"], errors="coerce")
for col in ["Predicted_Revenue", "Lower_95", "Upper_95"]:
    if col in forecast_df.columns:
        forecast_df[col] = pd.to_numeric(forecast_df[col], errors="coerce")
forecast_df = forecast_df.dropna(subset=["Date", "Predicted_Revenue"]).sort_values("Date")
forecast_df["Month_Name"] = forecast_df["Date"].dt.strftime("%b %Y")
forecast_df["Predicted_Lakh"] = forecast_df["Predicted_Revenue"] / 100000
if "Lower_95" in forecast_df.columns:
    forecast_df["Lower_Lakh"] = forecast_df["Lower_95"] / 100000
if "Upper_95" in forecast_df.columns:
    forecast_df["Upper_Lakh"] = forecast_df["Upper_95"] / 100000

for metric in ["MAE", "RMSE", "MAPE_Percent"]:
    if metric in model_df.columns:
        model_df[metric] = pd.to_numeric(model_df[metric], errors="coerce")

st.markdown("# 📈 Predictive Sales Analytics")
st.caption("Historical Revenue Analysis | Machine Learning | ARIMA Forecasting")

page = st.sidebar.radio(
    "Select Section",
    ["Overview", "Historical Analysis", "Model Performance", "Revenue Forecast", "Forecast Report"]
)
st.sidebar.title("Dashboard Navigation")
st.sidebar.caption("Dataset: Synthetic sales transactions")
st.sidebar.caption("Historical period: January–December 2025")

if page == "Overview":
    st.header("Business Overview")
    annual = monthly_df["Revenue"].sum()
    average = monthly_df["Revenue"].mean()
    highest = monthly_df.loc[monthly_df["Revenue"].idxmax()]
    a, b, c, d = st.columns(4)
    a.metric("Annual Revenue", f"₹{annual / 10000000:.2f} Cr")
    b.metric("Average Monthly Revenue", money_lakh(average))
    c.metric("Highest Monthly Revenue", money_lakh(highest["Revenue"]))
    d.metric("Forecast Period", f"{len(forecast_df)} Months")

    st.divider()
    st.subheader("Historical Revenue vs Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_df["Date"], y=monthly_df["Revenue_Lakh"],
                             mode="lines+markers", name="Historical Revenue",
                             line=dict(width=3, color="#0866C6")))
    if {"Lower_Lakh", "Upper_Lakh"}.issubset(forecast_df.columns):
        fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Upper_Lakh"],
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Lower_Lakh"],
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(255,99,99,.22)",
                                 name="95% Prediction Interval"))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted_Lakh"],
                             mode="lines+markers", name="ARIMA Forecast",
                             line=dict(width=3, color="#65B9F7", dash="dash")))
    fig.update_layout(template="plotly_white", height=500,
                      xaxis_title="Month", yaxis_title="Revenue (₹ Lakh)")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Historical data covers January–December 2025. See Revenue Forecast for forecast dates and intervals.")

elif page == "Historical Analysis":
    st.header("Historical Revenue Analysis")
    st.caption("Month-wise sales revenue analysis for January–December 2025")
    monthly = monthly_df.copy()
    highest = monthly.loc[monthly["Revenue"].idxmax()]
    lowest = monthly.loc[monthly["Revenue"].idxmin()]
    average = monthly["Revenue"].mean()
    a, b, c = st.columns(3)
    a.metric("Highest Revenue Month", highest["Month_Name"], money_lakh(highest["Revenue"]))
    b.metric("Lowest Revenue Month", lowest["Month_Name"], money_lakh(lowest["Revenue"]))
    c.metric("Average Monthly Revenue", money_lakh(average))
    st.divider()

    st.subheader("1. Monthly Revenue Trend")
    trend = px.line(monthly, x="Month_Name", y="Revenue_Lakh", markers=True,
                    title="Monthly Revenue Trend – 2025",
                    labels={"Month_Name": "Month", "Revenue_Lakh": "Revenue (₹ Lakh)"})
    trend.update_traces(line=dict(width=3), marker=dict(size=9))
    trend.update_layout(template="plotly_white", height=430)
    st.plotly_chart(trend, use_container_width=True)

    st.subheader("2. Month-wise Revenue Comparison")
    bars = px.bar(monthly, x="Month_Name", y="Revenue_Lakh", text="Revenue_Lakh",
                  title="Revenue Generated Every Month",
                  labels={"Month_Name": "Month", "Revenue_Lakh": "Revenue (₹ Lakh)"})
    bars.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    bars.update_layout(template="plotly_white", height=430)
    st.plotly_chart(bars, use_container_width=True)

    st.subheader("3. Monthly Revenue Details")
    display = monthly[["Month_Name", "Revenue_Lakh"]].copy()
    display.columns = ["Month", "Revenue (₹ Lakh)"]
    display["Revenue (₹ Lakh)"] = display["Revenue (₹ Lakh)"].round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button("Download Monthly Revenue Analysis",
                       data=display.to_csv(index=False).encode("utf-8"),
                       file_name="monthly_revenue_analysis.csv", mime="text/csv")

elif page == "Model Performance":
    st.header("Forecast Model Comparison")
    st.caption("Chronological test period: October–December 2025")
    required = {"Model", "MAE", "RMSE", "MAPE_Percent"}
    if not required.issubset(model_df.columns):
        st.error(f"model_comparison.csv needs these columns: {sorted(required)}")
        st.write("Found:", list(model_df.columns))
        st.stop()
    ranked = model_df.sort_values("MAE").copy()
    st.dataframe(ranked.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}",
                                      "MAPE_Percent": "{:.2f}%"}),
                 use_container_width=True, hide_index=True)
    best = ranked.iloc[0]
    a, b, c = st.columns(3)
    a.metric("Best Model by Test MAE", str(best["Model"]))
    b.metric("Test MAE", f"₹{best['MAE']:,.0f}")
    c.metric("Test MAPE", f"{best['MAPE_Percent']:.2f}%")

    st.subheader("Model Error Comparison")
    choice = st.selectbox("Choose evaluation metric", ["MAE", "RMSE", "MAPE_Percent"])
    metric_fig = px.bar(ranked, x="Model", y=choice, color="Model", text=choice,
                        title=f"Model Comparison – {choice}")
    metric_fig.update_layout(template="plotly_white", height=420, showlegend=False)
    st.plotly_chart(metric_fig, use_container_width=True)

    st.subheader("Actual vs Predicted Revenue")
    date_col = find_column(predictions_df, ["Date", "Order_Date", "Month"])
    actual_col = find_column(predictions_df, ["Actual", "Actual_Revenue", "Actual_Values", "y_test", "Revenue"])
    pred_col = find_column(predictions_df, ["Predicted", "Predicted_Revenue", "ARIMA_Predicted",
                                            "ARIMA_Prediction", "Prediction", "Forecast"])
    if actual_col and pred_col:
        plot_data = predictions_df.copy()
        if date_col:
            plot_data[date_col] = pd.to_datetime(plot_data[date_col], errors="coerce")
            xcol = date_col
        else:
            plot_data = plot_data.reset_index().rename(columns={"index": "Test Month"})
            xcol = "Test Month"
        plot_data[actual_col] = pd.to_numeric(plot_data[actual_col], errors="coerce")
        plot_data[pred_col] = pd.to_numeric(plot_data[pred_col], errors="coerce")
        plot_data = plot_data.dropna(subset=[actual_col, pred_col])
        melted = plot_data[[xcol, actual_col, pred_col]].melt(
            id_vars=xcol, var_name="Series", value_name="Revenue")
        actual_fig = px.line(melted, x=xcol, y="Revenue", color="Series",
                             markers=True, title="Actual and Predicted Revenue")
        actual_fig.update_layout(template="plotly_white", height=420)
        st.plotly_chart(actual_fig, use_container_width=True)
    else:
        st.warning("Could not identify actual/predicted columns automatically in test_predictions.csv.")
        st.write("Available columns:", list(predictions_df.columns))
        st.dataframe(predictions_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Model Evaluation Insights")
    st.markdown(f"""
**Best-performing model on this test period: {best['Model']}**

- **MAE:** ₹{best['MAE']:,.0f} – average absolute prediction error.
- **RMSE:** ₹{best['RMSE']:,.0f} – gives greater weight to larger errors.
- **MAPE:** {best['MAPE_Percent']:.2f}% – average absolute percentage error.

The evaluation uses only three test months (October–December 2025).
Treat these results as performance on this limited test period, not proof
of generalization to all future sales conditions.
""")

elif page == "Revenue Forecast":
    st.header("Revenue Forecast")
    st.caption("ARIMA(1,1,0) forecast generated from the 2025 monthly revenue series")
    a, b, c = st.columns(3)
    a.metric("Forecast Months", str(len(forecast_df)))
    b.metric("Average Forecast Revenue", money_lakh(forecast_df["Predicted_Revenue"].mean()))
    c.metric("Total Forecast Revenue", money_lakh(forecast_df["Predicted_Revenue"].sum()))

    st.subheader("Monthly Revenue Forecast")
    fig = go.Figure()
    if {"Lower_Lakh", "Upper_Lakh"}.issubset(forecast_df.columns):
        fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Upper_Lakh"],
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Lower_Lakh"],
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(255,99,99,.22)",
                                 name="95% Prediction Interval"))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted_Lakh"],
                             mode="lines+markers", name="ARIMA Forecast",
                             line=dict(width=3, color="#0866C6")))
    fig.update_layout(template="plotly_white", height=450,
                      xaxis_title="Forecast Month", yaxis_title="Revenue (₹ Lakh)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Values and 95% Prediction Intervals")
    cols = ["Month_Name", "Predicted_Revenue"]
    rename = {"Month_Name": "Forecast Month", "Predicted_Revenue": "Predicted Revenue (₹)"}
    if "Lower_95" in forecast_df.columns:
        cols.append("Lower_95")
        rename["Lower_95"] = "Lower 95% (₹)"
    if "Upper_95" in forecast_df.columns:
        cols.append("Upper_95")
        rename["Upper_95"] = "Upper 95% (₹)"
    table = forecast_df[cols].rename(columns=rename)
    st.dataframe(table.style.format({col: "₹{:,.0f}" for col in table.columns
                                     if col != "Forecast Month"}),
                 use_container_width=True, hide_index=True)
    first_date = forecast_df["Date"].min()
    st.warning(f"This forecast begins in {first_date:%B %Y}. If this date is already past, "
               "treat it as a historical forecasting experiment, not a current future projection.")
    st.info("Only 12 monthly observations from 2025 were used. Forecast intervals are estimates; "
            "additional historical data is needed for stronger reliability.")

elif page == "Forecast Report":
    st.header("Forecast Report")
    st.caption("Summary of historical revenue, model metrics and forecast results")
    total = monthly_df["Revenue"].sum()
    avg = monthly_df["Revenue"].mean()
    high = monthly_df.loc[monthly_df["Revenue"].idxmax()]
    low = monthly_df.loc[monthly_df["Revenue"].idxmin()]
    best_model = model_df.loc[model_df["MAE"].idxmin(), "Model"]
    summary = pd.DataFrame({
        "Measure": ["Historical Period", "Total Annual Revenue", "Average Monthly Revenue",
                    "Highest Revenue Month", "Highest Monthly Revenue", "Lowest Revenue Month",
                    "Lowest Monthly Revenue", "Best Model by Test MAE"],
        "Result": [f"{monthly_df['Date'].min():%b %Y} – {monthly_df['Date'].max():%b %Y}",
                   f"₹{total:,.2f}", f"₹{avg:,.2f}", high["Month_Name"],
                   f"₹{high['Revenue']:,.2f}", low["Month_Name"],
                   f"₹{low['Revenue']:,.2f}", str(best_model)]
    })
    st.subheader("Historical Revenue Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.subheader("Model Comparison")
    st.dataframe(model_df, use_container_width=True, hide_index=True)
    st.subheader("Revenue Forecast")
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    st.download_button("Download Forecast Report (CSV)",
                       data=forecast_df.to_csv(index=False).encode("utf-8"),
                       file_name="predictive_sales_forecast_report.csv", mime="text/csv")
    report = (f"PREDICTIVE SALES ANALYTICS REPORT\n"
              f"Historical data: January–December 2025\n"
              f"Total annual revenue: ₹{total:,.2f}\n"
              f"Average monthly revenue: ₹{avg:,.2f}\n"
              f"Highest month: {high['Month_Name']} (₹{high['Revenue']:,.2f})\n"
              f"Lowest month: {low['Month_Name']} (₹{low['Revenue']:,.2f})\n"
              f"Best model by test MAE: {best_model}\n"
              "Note: evaluation uses a three-month test period; the dataset is synthetic.\n")
    st.download_button("Download Executive Summary (TXT)", data=report,
                       file_name="predictive_sales_executive_summary.txt", mime="text/plain")

st.sidebar.divider()
st.sidebar.caption("Predictive Sales Analytics | Educational prototype")
