
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# 1. Page configuration
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌍",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
DATA_PATH = BASE_DIR / "historical_aqi_data.csv"
FEATURE_PATH = BASE_DIR / "feature_columns.txt"
METRICS_PATH = BASE_DIR / "model_metrics.csv"

REQUIRED_POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]

# 2. Load project files
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data

@st.cache_data
def load_features():
    with open(FEATURE_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH)

try:
    model = load_model()
    df = load_data()
    feature_columns = load_features()
    metrics = load_metrics()
except Exception as exc:
    st.error(f"Could not load project files: {exc}")
    st.stop()

# 3. Check feature order
expected_features = [
    "pm25", "pm10", "no2", "so2", "co", "o3",
    "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
    "aqi_lag_7", "aqi_7day_avg", "month", "day_of_week"
]

if feature_columns != expected_features:
    st.error("Feature columns do not match the model's expected order.")
    st.stop()

required_columns = [
    "station_id", "date", "aqi", *REQUIRED_POLLUTANTS
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()

# 4. Feature engineering
@st.cache_data
def prepare_historical_features(data):
    data = data.copy()
    data = data.sort_values(
        ["station_id", "date"]
    ).reset_index(drop=True)

    grouped_aqi = data.groupby("station_id")["aqi"]

    data["aqi_lag_1"] = grouped_aqi.shift(1)
    data["aqi_lag_2"] = grouped_aqi.shift(2)
    data["aqi_lag_3"] = grouped_aqi.shift(3)
    data["aqi_lag_7"] = grouped_aqi.shift(7)

    data["aqi_7day_avg"] = (
        data.groupby("station_id")["aqi"]
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=7
            ).mean()
        )
    )

    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.dayofweek

    return data

featured_df = prepare_historical_features(df)

model_inputs = [
    *REQUIRED_POLLUTANTS,
    "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
    "aqi_lag_7", "aqi_7day_avg",
    "month", "day_of_week"
]

eligible_df = featured_df.dropna(
    subset=model_inputs
).copy()

# 5. AQI category
def get_aqi_category(value):
    if value <= 50:
        return "Good"
    elif value <= 100:
        return "Satisfactory"
    elif value <= 200:
        return "Moderate"
    elif value <= 300:
        return "Poor"
    elif value <= 400:
        return "Very Poor"
    else:
        return "Severe"

# 6. Dashboard heading
st.title("🌍 Air Quality Index Prediction")
st.caption(
    "Random Forest-based next-day AQI prediction dashboard"
)

with st.expander("About this application"):
    st.write(
        "This application uses historical air quality data "
        "and a trained Random Forest model to predict next-day AQI."
    )
    st.info(
        "Select a measurement date. The application predicts "
        "AQI for the following calendar day."
    )

# 7. Station selection
st.subheader("1. Select Monitoring Station")

if eligible_df.empty:
    st.error("No eligible records are available.")
    st.stop()

station_ids = sorted(
    eligible_df["station_id"].dropna().unique().tolist()
)

selected_station = st.selectbox(
    "Choose a station",
    station_ids
)

station_rows = eligible_df[
    eligible_df["station_id"] == selected_station
].sort_values("date")

available_dates = station_rows["date"].dt.date.tolist()

if not available_dates:
    st.warning("No eligible dates found for this station.")
    st.stop()

selected_date = st.selectbox(
    "Select measurement date",
    available_dates,
    index=len(available_dates) - 1,
    format_func=lambda d: d.strftime("%d %B %Y")
)

selected_row = station_rows[
    station_rows["date"].dt.date == selected_date
].iloc[0]

prediction_date = (
    pd.Timestamp(selected_date) + pd.Timedelta(days=1)
)

# 8. Display pollutant measurements
st.subheader("2. Pollutant Measurements")

pollutant_labels = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2": "NO₂",
    "so2": "SO₂",
    "co": "CO",
    "o3": "O₃"
}

cols = st.columns(3)

for i, pollutant in enumerate(REQUIRED_POLLUTANTS):
    with cols[i % 3]:
        st.metric(
            pollutant_labels[pollutant],
            f"{selected_row[pollutant]:.2f}"
        )

st.caption(
    f"Recorded AQI: {selected_row['aqi']:.1f}"
)

# 9. Prepare model input
input_values = {
    feature: selected_row[feature]
    for feature in feature_columns
}

input_df = pd.DataFrame(
    [input_values],
    columns=feature_columns
)

# 10. Make prediction
st.subheader("3. Predict Next-Day AQI")

if st.button(
    "Predict Next-Day AQI",
    type="primary",
    use_container_width=True
):
    try:
        prediction = float(model.predict(input_df)[0])
        category = get_aqi_category(prediction)

        st.session_state["last_prediction"] = {
            "station": selected_station,
            "measurement_date": selected_date,
            "prediction_date": prediction_date,
            "prediction": prediction,
            "category": category
        }

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

# 11. Display prediction result
if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]

    if (
        result["station"] == selected_station
        and result["measurement_date"] == selected_date
    ):
        st.subheader("4. Prediction Result")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicted AQI",
            f"{result['prediction']:.2f}"
        )

        c2.metric(
            "AQI Category",
            result["category"]
        )

        c3.metric(
            "Prediction Date",
            result["prediction_date"].strftime("%d %b %Y")
        )

        st.success(
            f"Predicted AQI: {result['prediction']:.2f} "
            f"({result['category']})"
        )

        if not 0 <= result["prediction"] <= 500:
            st.warning(
                "The prediction is outside the usual 0–500 AQI scale."
            )

# 12. Model performance
st.divider()
st.subheader("5. Model Performance")

metric_lookup = {}

if {"Metric", "Value"}.issubset(metrics.columns):
    metric_lookup = {
        str(row["Metric"]).upper(): float(row["Value"])
        for _, row in metrics.iterrows()
        if pd.notna(row["Value"])
    }

mae, rmse, r2 = st.columns(3)

if "MAE" in metric_lookup:
    mae.metric("MAE", f"{metric_lookup['MAE']:.2f}")

if "RMSE" in metric_lookup:
    rmse.metric("RMSE", f"{metric_lookup['RMSE']:.2f}")

if "R2" in metric_lookup:
    r2.metric("R² Score", f"{metric_lookup['R2']:.4f}")

with st.expander("View complete model metrics"):
    st.dataframe(metrics, use_container_width=True)

# 13. Historical AQI trend
st.divider()
st.subheader("6. Historical AQI Trend")

recent_history = featured_df[
    (featured_df["station_id"] == selected_station)
    & (featured_df["date"].dt.date <= selected_date)
][["date", "aqi"]].dropna().tail(30)

if not recent_history.empty:
    recent_history = recent_history.set_index("date")
    st.line_chart(recent_history, y="aqi")
else:
    st.info("No historical AQI values available.")

# 14. Footer
st.divider()

st.caption(
    "AQI Prediction System | Random Forest Regressor"
)
