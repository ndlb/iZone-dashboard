import io
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from prophet import Prophet

REQUIRED_COLS = {"Timestamp", "Average Occupancy"}


# ---------------------------------------------------------------------------
# Custom events (academic calendar: breaks + exam periods)
# ---------------------------------------------------------------------------

def build_custom_events():
    events = pd.DataFrame([
        # 2026
        {"holiday": "spring_break",  "start": "2026-03-07", "end": "2026-03-15"},
        {"holiday": "reading_days",  "start": "2026-05-02", "end": "2026-05-04"},
        {"holiday": "finals_week",   "start": "2026-05-05", "end": "2026-05-10"},
        # 2025
        {"holiday": "spring_break",  "start": "2025-03-08", "end": "2025-03-16"},
        {"holiday": "reading_days",  "start": "2025-05-03", "end": "2025-05-05"},
        {"holiday": "finals_week",   "start": "2025-05-06", "end": "2025-05-11"},
        {"holiday": "summer_break",  "start": "2025-05-18", "end": "2025-08-24"},
        {"holiday": "fall_break",    "start": "2025-10-13", "end": "2025-10-14"},
        {"holiday": "thanksgiving",  "start": "2025-11-26", "end": "2025-11-30"},
        {"holiday": "reading_days",  "start": "2025-12-09", "end": "2025-12-11"},
        {"holiday": "finals_week",   "start": "2025-12-12", "end": "2025-12-17"},
        {"holiday": "winter_break",  "start": "2025-12-17", "end": "2026-01-19"},
        # 2024
        {"holiday": "spring_break",  "start": "2024-03-09", "end": "2024-03-17"},
        {"holiday": "reading_days",  "start": "2024-05-01", "end": "2024-05-04"},
        {"holiday": "finals_week",   "start": "2024-05-05", "end": "2024-05-11"},
        {"holiday": "summer_break",  "start": "2024-05-12", "end": "2024-08-25"},
        {"holiday": "fall_break",    "start": "2024-10-14", "end": "2024-10-15"},
        {"holiday": "thanksgiving",  "start": "2024-11-27", "end": "2024-12-01"},
        {"holiday": "reading_days",  "start": "2024-12-10", "end": "2024-12-12"},
        {"holiday": "finals_week",   "start": "2024-12-13", "end": "2024-12-18"},
        {"holiday": "winter_break",  "start": "2024-12-19", "end": "2025-01-20"},
        # 2023
        {"holiday": "summer_break",  "start": "2023-05-12", "end": "2023-08-29"},
        {"holiday": "fall_break",    "start": "2023-10-16", "end": "2023-10-17"},
        {"holiday": "thanksgiving",  "start": "2023-11-22", "end": "2023-11-26"},
        {"holiday": "reading_days",  "start": "2023-12-14", "end": "2023-12-16"},
        {"holiday": "finals_week",   "start": "2023-12-17", "end": "2023-12-22"},
        {"holiday": "winter_break",  "start": "2023-12-23", "end": "2024-01-16"},
    ])
    events["start"] = pd.to_datetime(events["start"])
    events["end"]   = pd.to_datetime(events["end"])

    custom_events = pd.DataFrame({
        "holiday":      events["holiday"],
        "ds":           events["start"],
        "lower_window": 0,
        "upper_window": (events["end"] - events["start"]).dt.days,
    })
    return custom_events


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

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

    df = try_read(skiprows=None)
    if df is not None and REQUIRED_COLS.issubset(df.columns):
        pass
    else:
        df = try_read(skiprows=6)
        if df is None or not REQUIRED_COLS.issubset(df.columns):
            raise ValueError(
                "Could not find required columns 'Timestamp' and "
                "'Average Occupancy' in the CSV (with or without skipping "
                "the first 6 lines)."
            )

    df = df[list(REQUIRED_COLS)].dropna()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")
    df = df.rename(columns={"Timestamp": "ds", "Average Occupancy": "y"})
    return df


# ---------------------------------------------------------------------------
# Training window: most recent 720 days
# ---------------------------------------------------------------------------

def trim_to_training_window(df, training_days=720):
    """
    Keep only the most recent `training_days` days of data.
    This ensures Prophet trains on a focused, recent window rather than
    all historical data, which improves forecast relevance.
    """
    cutoff = df["ds"].max() - pd.Timedelta(days=training_days)
    trimmed = df[df["ds"] >= cutoff].copy()
    return trimmed


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_prophet(df, custom_events):
    """
    Train a Prophet model with:
      - daily_seasonality = True
      - weekly_seasonality = True
      - yearly_seasonality = False   (per spec)
      - custom academic-calendar holidays injected via the holidays DataFrame
    """
    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        holidays=custom_events,
    )
    m.fit(df)
    return m


# ---------------------------------------------------------------------------
# Forecasting helpers
# ---------------------------------------------------------------------------

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
    daily["date"]         = pd.to_datetime(daily["date"])
    daily["dow"]          = daily["date"].dt.weekday        # 0=Mon, 6=Sun
    daily["day_name"]     = daily["date"].dt.day_name()
    daily["dom"]          = daily["date"].dt.day            # day of month
    daily["week_in_month"] = ((daily["dom"] - 1) // 7) + 1
    return daily


# ---------------------------------------------------------------------------
# Calendar heatmap helpers
# ---------------------------------------------------------------------------

def build_calendar_matrix(daily_df):
    """
    Build matrices for the heatmap:

      rows  = calendar weeks starting from the first date in the selected range
      cols  = day of week (0=Mon..6=Sun)

    Reading left→right, top→bottom (ignoring blanks) gives dates in order.

    Returns:
      mat         - traffic values [week_index, dow]
      day_labels  - ['Mon', ..., 'Sun']  (for x-axis)
      date_labels - same shape as mat, with 'dd/mm' strings for annotation
    """
    if daily_df.empty:
        return None, None, None

    daily_df = daily_df.sort_values("date").copy()

    start_date    = daily_df["date"].min()
    start_weekday = start_date.weekday()  # 0=Mon..6=Sun

    daily_df["offset_days"]  = (daily_df["date"] - start_date).dt.days
    daily_df["global_index"] = start_weekday + daily_df["offset_days"]
    daily_df["week_index"]   = (daily_df["global_index"] // 7).astype(int)
    daily_df["dow"]          = (daily_df["global_index"] %  7).astype(int)

    max_week    = daily_df["week_index"].max()
    mat         = np.full((max_week + 1, 7), np.nan)
    date_labels = np.full((max_week + 1, 7), "", dtype=object)

    for _, row in daily_df.iterrows():
        r = row["week_index"]
        c = row["dow"]
        mat[r, c]         = row["traffic_pred"]
        date_labels[r, c] = row["date"].strftime("%m/%d")

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return mat, day_labels, date_labels


def plot_calendar_heatmap(daily_df, title):
    """
    GitHub-style monthly heatmap:
      x-axis: day of week (Mon–Sun)
      y-axis: weeks (no labels shown)
      color: occupancy (green low -> red high)
      values inside each box: date as 'dd/mm'
    """
    mat, day_labels, date_labels = build_calendar_matrix(daily_df)
    if mat is None:
        return None

    fig, ax = plt.subplots(figsize=(8, 3))

    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="RdYlGn_r")

    ax.set_xticks(range(len(day_labels)))
    ax.set_xticklabels(day_labels)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([])
    ax.set_xlabel("Day of week")
    ax.set_ylabel("")
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isnan(val) and date_labels[i, j] != "":
                ax.text(
                    j, i,
                    date_labels[i, j],
                    ha="center", va="center",
                    fontsize=8, color="black",
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
    time_labels = day_df["ds"].dt.strftime("%H:%M")
    traffic = day_df["yhat"].values

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(traffic)), traffic, marker="o")

    # Show a tick every 2 hours (every 4th 30-min slot) to avoid crowding
    tick_every = 4
    tick_positions = range(0, len(time_labels), tick_every)
    ax.set_xticks(list(tick_positions))
    ax.set_xticklabels([time_labels.iloc[i] for i in tick_positions], rotation=45, ha="right")

    ax.set_xlabel("Time of day")
    ax.set_ylabel("Predicted avg occupancy")
    ax.set_title(title)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def main():
    st.title("Traffic Forecast Dashboard (Prophet)")
    st.write(
        "Upload a CSV with **Timestamp** and **Average Occupancy** "
        "(first 6 lines of metadata are okay). "
        "Choose a ~1-month range and how you want to view the prediction."
    )

    # Pre-build custom events once
    custom_events = build_custom_events()

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is None:
        st.stop()

    # ---- Load CSV ----
    try:
        df_full = load_csv_with_optional_header(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    st.success(
        f"Loaded {len(df_full):,} rows from CSV. "
        f"Data range: {df_full['ds'].min().date()} → {df_full['ds'].max().date()}"
    )

    # ---- Trim to training window ----
    TRAINING_DAYS = 720
    df_train = trim_to_training_window(df_full, training_days=TRAINING_DAYS)

    st.info(
        f"Training window: most recent **{TRAINING_DAYS} days** of data — "
        f"{df_train['ds'].min().date()} → {df_train['ds'].max().date()} "
        f"({len(df_train):,} rows used for training)."
    )

    # ---- Date range selector ----
    min_date     = df_train["ds"].min().date()
    max_date     = df_train["ds"].max().date()
    default_end   = max_date
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

    # ---- View mode ----
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

    # ---- Run forecast ----
    if st.button("Run forecast"):
        with st.spinner(
            "Training Prophet model on the most recent 720 days and generating forecast…"
        ):
            try:
                model = train_prophet(df_train, custom_events)
                last_ds = df_train["ds"].max()

                start_dt = dt.datetime.combine(start_date, dt.time(0, 0))
                end_dt   = dt.datetime.combine(end_date,   dt.time(23, 30))

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

                mask   = forecast["ds"].dt.date == detail_date
                day_df = forecast.loc[mask].copy()

                if day_df.empty:
                    st.error("No forecast data for that detail date.")
                    st.stop()

                st.subheader(f"Predicted half-hourly occupancy for {detail_date}")
                table = day_df[["ds", "yhat"]].copy()
                table["ds"] = table["ds"].dt.strftime("%Y-%m-%d %H:%M")
                table = table.rename(
                    columns={"ds": "timestamp", "yhat": "predicted_avg_occupancy"}
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
