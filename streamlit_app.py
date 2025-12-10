import io
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from prophet import Prophet

REQUIRED_COLS = {"Timestamp", "Average Occupancy"}


def load_csv_with_optional_header(uploaded_file):
    """
    Load CSV that may or may not have 6 lines of metadata at the top.

    Strategy:
      1) Try read_csv with no skiprows.
      2) If parsing fails OR required columns not found, try again with skiprows=6.
      3) Use the first version that contains both 'Timestamp' and 'Average Occupancy'.

    Uses only 'Timestamp' + 'Average Occupancy'.
    """
    content = uploaded_file.getvalue()

    def try_read(skiprows=None):
        try:
            if skiprows is None:
                df_ = pd.read_csv(io.BytesIO(content))
            else:
                df_ = pd.read_csv(io.BytesIO(content), skiprows=skiprows)
        except Exception:
            return None
        return df_

    # First: try as-is
    df = try_read(skiprows=None)
    if df is not None and REQUIRED_COLS.issubset(df.columns):
        pass  # df is good
    else:
        # Second: try skipping first 6 lines (metadata style)
        df = try_read(skiprows=6)
        if df is None or not REQUIRED_COLS.issubset(df.columns):
            # Neither attempt produced the expected columns
            raise ValueError(
                "Could not find required columns 'Timestamp' and "
                "'Average Occupancy' in the CSV (with or without skipping "
                "the first 6 lines)."
            )

    # At this point df has the right columns
    df = df[list(REQUIRED_COLS)].dropna()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")
    df = df.rename(columns={"Timestamp": "ds", "Average Occupancy": "y"})
    return df



def train_prophet(df):
    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    m.fit(df)
    return m


def forecast_range(model, last_ds, start_date, end_date, freq="30min"):
    """
    Make predictions from start_date to end_date (inclusive),
    at 30-minute frequency, extending beyond last_ds if needed.
    """
    if end_date > last_ds:
        horizon = int(((end_date - last_ds).total_seconds() // (30 * 60)))
    else:
        horizon = 0

    future = model.make_future_dataframe(periods=horizon, freq=freq)
    forecast = model.predict(future)

    mask = (forecast["ds"] >= start_date) & (forecast["ds"] <= end_date)
    return forecast.loc[mask].copy()


def aggregate_to_daily_avg(forecast_df):
    """
    Aggregate half-hourly predictions into daily *average* occupancy.
    Change .mean() to .sum() if you prefer daily totals.
    """
    forecast_df["date"] = forecast_df["ds"].dt.date
    daily = (
        forecast_df
        .groupby("date")["yhat"]
        .mean()
        .reset_index()
        .rename(columns={"yhat": "traffic_pred"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily["dow"] = daily["date"].dt.weekday        # 0=Mon, 6=Sun
    daily["day_name"] = daily["date"].dt.day_name()
    daily["dom"] = daily["date"].dt.day            # day of month
    daily["week_in_month"] = ((daily["dom"] - 1) // 7) + 1
    return daily


def build_calendar_matrix(daily_df):
    """
    Build matrices for the heatmap:

      rows  = weeks in the month (1..max_week_in_month)
      cols  = day of week (0=Mon..6=Sun)

    Returns:
      mat       - traffic values [weeks, dow]
      day_labels - ['Mon', ..., 'Sun']  (for x-axis)
      date_labels - same shape as mat, with day-of-month strings for annotation
    """
    if daily_df.empty:
        return None, None, None

    max_week = daily_df["week_in_month"].max()

    # traffic matrix and date matrix (day-of-month as string)
    mat = np.full((max_week, 7), np.nan)
    date_labels = np.full((max_week, 7), "", dtype=object)

    for _, row in daily_df.iterrows():
        r = row["week_in_month"] - 1   # week index
        c = row["dow"]                 # 0..6 -> Mon..Sun
        mat[r, c] = row["traffic_pred"]
        date_labels[r, c] = str(row["dom"])

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return mat, day_labels, date_labels


def plot_calendar_heatmap(daily_df, title):
    """
    GitHub-style monthly heatmap:
      x-axis: day of week (Mon–Sun)
      y-axis: weeks (no labels shown)
      color: occupancy (green low -> red high)
      values inside each box: day-of-month number
    """
    mat, day_labels, date_labels = build_calendar_matrix(daily_df)
    if mat is None:
        return None

    fig, ax = plt.subplots(figsize=(8, 3))

    # Colormap: RdYlGn_r (green low, yellow mid, red high)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="RdYlGn_r")

    # X-axis: day of week
    ax.set_xticks(range(len(day_labels)))
    ax.set_xticklabels(day_labels)

    # Y-axis: weeks, but we don't show labels
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([])

    ax.set_xlabel("Day of week")
    ax.set_ylabel("")  # no label needed
    ax.set_title(title)

    # Annotate each non-NaN cell with the day-of-month
    for i in range(mat.shape[0]):          # weeks
        for j in range(mat.shape[1]):      # days of week
            val = mat[i, j]
            if not np.isnan(val) and date_labels[i, j] != "":
                ax.text(
                    j, i,
                    date_labels[i, j],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Predicted avg occupancy")

    plt.tight_layout()
    return fig


def plot_daily_line(day_df, title):
    """
    Line chart for a single day:
      x-axis: time (timestamp)
      y-axis: predicted avg occupancy
    """
    day_df = day_df.sort_values("ds").copy()
    times = day_df["ds"]
    traffic = day_df["yhat"].values

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, traffic, marker="o")

    ax.set_xlabel("Time of day")
    ax.set_ylabel("Predicted avg occupancy")
    ax.set_title(title)
    fig.autofmt_xdate()  # rotate x labels

    plt.tight_layout()
    return fig


def main():
    st.title("Traffic Forecast Dashboard (Prophet)")
    st.write(
        "Upload a CSV with **Timestamp** and **Average Occupancy** "
        "(first 6 lines of metadata are okay). "
        "Choose a ~1-month range and how you want to view the prediction."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is None:
        st.stop()

    # Load CSV and show basic info
    try:
        df = load_csv_with_optional_header(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    st.success(
        f"Loaded {len(df):,} rows from CSV. "
        f"Data range: {df['ds'].min().date()} → {df['ds'].max().date()}"
    )

    # Suggested default range: last 30 days of data if possible
    min_date = df["ds"].min().date()
    max_date = df["ds"].max().date()
    default_end = max_date
    default_start = max(min_date, default_end - dt.timedelta(days=29))

    date_range = st.date_input(
        "Forecast date range (should be about one month)",
        value=(default_start, default_end),
    )

    if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
        st.error("Please select a start and end date.")
        st.stop()

    start_date, end_date = date_range
    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    diff_days = (end_date - start_date).days + 1
    st.write(f"Selected range: {start_date} → {end_date} ({diff_days} days)")

    if diff_days < 25 or diff_days > 35:
        st.warning(
            "For stability, the forecasting range should be roughly one month "
            "(25–35 days). You can still run it, but results may be odd."
        )

    summary_level = st.radio(
        "Summary level",
        ["Monthly heatmap", "Single-day line chart"],
        index=0,
    )

    detail_date = None
    if summary_level == "Single-day line chart":
        detail_date = st.date_input(
            "Detail date (must lie inside the range above)",
            value=start_date,
        )

    if st.button("Run forecast"):
        with st.spinner("Training Prophet model and generating forecast..."):
            try:
                model = train_prophet(df)
                last_ds = df["ds"].max()

                start_dt = dt.datetime.combine(start_date, dt.time(0, 0))
                end_dt = dt.datetime.combine(end_date, dt.time(23, 30))

                forecast = forecast_range(model, last_ds, start_dt, end_dt)
            except Exception as e:
                st.error(f"Error during forecasting: {e}")
                st.stop()

            if forecast.empty:
                st.error("No forecast generated for this range.")
                st.stop()

            if summary_level == "Monthly heatmap":
                daily = aggregate_to_daily_avg(forecast)
                st.subheader("Daily predicted average occupancy")
                st.dataframe(
                    daily[["date", "traffic_pred"]]
                        .round(2)
                        .rename(columns={"traffic_pred": "predicted_avg_occupancy"})
                )

                fig = plot_calendar_heatmap(
                    daily,
                    title=f"Predicted avg occupancy ({start_date} → {end_date})",
                )
                if fig is None:
                    st.error("Could not build calendar heatmap.")
                else:
                    st.subheader("Monthly heatmap (GitHub-style)")
                    st.pyplot(fig)

            else:  # Single-day line chart
                if not (start_date <= detail_date <= end_date):
                    st.error("Detail date must lie inside the forecast range.")
                    st.stop()

                mask = forecast["ds"].dt.date == detail_date
                day_df = forecast.loc[mask].copy()

                if day_df.empty:
                    st.error("No forecast data for that detail date.")
                    st.stop()

                st.subheader(f"Predicted half-hourly occupancy for {detail_date}")
                table = day_df[["ds", "yhat"]].copy()
                table["ds"] = table["ds"].dt.strftime("%Y-%m-%d %H:%M")
                table = table.rename(
                    columns={
                        "ds": "timestamp",
                        "yhat": "predicted_avg_occupancy",
                    }
                ).round(2)
                st.dataframe(table)

                fig = plot_daily_line(
                    day_df,
                    title=f"Predicted avg occupancy on {detail_date}",
                )
                st.subheader("Daily line chart")
                st.pyplot(fig)


if __name__ == "__main__":
    main()
