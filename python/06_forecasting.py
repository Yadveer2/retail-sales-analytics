"""
06_forecasting.py

We use Facebook Prophet to forecast the next 6 months of revenue.

Prophet is a library built for time-series forecasting.
It takes a table with two columns:
  ds — the date
  y  — the value to forecast (revenue in our case)

It automatically detects trends and seasonal patterns.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings("ignore")

# --- paths ---
ROOT       = Path(__file__).resolve().parents[1]
DATA_FILE  = ROOT / "data" / "master.csv"
OUTPUT_DIR = ROOT / "docs" / "forecast"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Run 01_data_cleaning.ipynb first. File not found: {DATA_FILE}")
    return pd.read_csv(DATA_FILE, parse_dates=["order_purchase_timestamp"])


# ── 2. Aggregate to monthly revenue ──────────────────────────────────────────

def make_monthly_series(df):
    """
    Collapse the row-level sales data into one row per month.
    Prophet requires column names 'ds' (date) and 'y' (value).
    We only use 2017-01 to 2018-08 because the data outside that range is incomplete.
    """
    monthly = (
        df.groupby(pd.Grouper(key="order_purchase_timestamp", freq="MS"))["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"order_purchase_timestamp": "ds", "revenue": "y"})
    )

    # keep only the months with complete data
    monthly = monthly[
        (monthly["ds"] >= "2017-01-01") &
        (monthly["ds"] <= "2018-08-01")
    ].copy()

    return monthly


# ── 3. Train the forecast model ───────────────────────────────────────────────

def train_model(monthly):
    """
    Train Prophet on the monthly data.
    We turn off weekly and daily seasonality because we're working with monthly data.
    changepoint_prior_scale=0.1 makes the trend less wiggly (more stable).
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
    )
    model.fit(monthly)
    return model


# ── 4. Measure how accurate the model is ─────────────────────────────────────

def measure_accuracy(monthly, forecast):
    """
    MAPE = Mean Absolute Percentage Error.
    It tells us on average how far off our predictions are in %.
    Lower is better. < 10% is generally good.
    """
    # get the forecast values that match the historical dates
    in_sample = forecast[forecast["ds"] <= monthly["ds"].max()]["yhat"].values
    mape = mean_absolute_percentage_error(monthly["y"], in_sample) * 100
    return round(mape, 2)


# ── 5. Chart ──────────────────────────────────────────────────────────────────

def plot_forecast(monthly, forecast):
    """
    One chart: actual revenue (solid line) + forecast (dashed line) + confidence band.
    The vertical red line marks where the forecast starts.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # actual historical data
    ax.plot(monthly["ds"], monthly["y"],
            "o-", color="#2E86AB", label="Actual", linewidth=2)

    # forecast line
    ax.plot(forecast["ds"], forecast["yhat"],
            "--", color="#F18F01", label="Forecast", linewidth=2)

    # shaded area = 95% confidence interval
    ax.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        color="#F18F01", alpha=0.2, label="95% confidence band",
    )

    # vertical line showing where historical data ends
    ax.axvline(x=monthly["ds"].max(), color="red", linestyle=":", alpha=0.6, label="Forecast start")

    ax.set_title("Monthly Revenue — 6-Month Forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (BRL)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forecast_chart.png", dpi=150)
    plt.close()


# ── 6. Print summary ─────────────────────────────────────────────────────────

def print_summary(monthly, forecast, mape):
    # get only the 6 future months (rows after the last historical date)
    future = forecast[forecast["ds"] > monthly["ds"].max()].head(6)

    last_actual    = monthly["y"].iloc[-1]
    next_3m_avg    = future["yhat"].head(3).mean()
    next_6m_total  = future["yhat"].sum()

    print("\n--- Forecast Summary ---")
    print(f"Last actual month revenue : Rs {last_actual:>12,.2f}")
    print(f"Next 3-month avg forecast : Rs {next_3m_avg:>12,.2f}")
    print(f"Next 6-month total        : Rs {next_6m_total:>12,.2f}")
    print(f"Model accuracy (MAPE)     : {mape}%  (lower is better)")

    print("\nNext 6 months:")
    print(future[["ds", "yhat", "yhat_lower", "yhat_upper"]].round(2).to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df):,} rows")

    monthly = make_monthly_series(df)
    print(f"Monthly series: {len(monthly)} months ({monthly['ds'].min().date()} to {monthly['ds'].max().date()})")

    model = train_model(monthly)
    print("Model trained.")

    # generate dates for the next 6 months beyond the training data
    future   = model.make_future_dataframe(periods=6, freq="MS")
    forecast = model.predict(future)

    # save full forecast table
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
        OUTPUT_DIR / "forecast_6_months.csv", index=False
    )

    plot_forecast(monthly, forecast)
    print("Chart saved.")

    mape = measure_accuracy(monthly, forecast)
    print_summary(monthly, forecast, mape)

    print(f"\nDone. Outputs saved to: {OUTPUT_DIR.relative_to(ROOT)}")
