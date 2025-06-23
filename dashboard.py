import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import xgboost as xgb

# Sidebar file uploader
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your power consumption CSV", type=["csv"])

# Load and clean data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, sep=';', header=None)
    df = df[0].str.split(',', expand=True)
    df.columns = [
        'Date', 'Time',
        'Global_active_power', 'Global_reactive_power',
        'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    df = df[df['Date'] != 'Date']
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    for col in [
        'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    df.set_index('Datetime', inplace=True)
    return df

if uploaded_file:
    df = load_data(uploaded_file)
    st.success("✅ Data uploaded and loaded successfully.")
else:
    st.warning("Please upload a valid dataset to continue.")
    st.stop()

# Resample data
hourly = df.resample('H').mean(numeric_only=True)
daily = df.resample('D').sum(numeric_only=True)

# Train Forecasting model
@st.cache_resource
def train_forecast_model(hourly_df):
    hr = hourly_df.copy()
    hr['hour'] = hr.index.hour
    hr['dayofweek'] = hr.index.dayofweek
    hr['month'] = hr.index.month
    hr['target'] = hr['Global_active_power'].shift(-1)
    hr.dropna(inplace=True)
    X = hr[['Global_active_power', 'hour', 'dayofweek', 'month']]
    y = hr['target']
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, verbosity=0)
    model.fit(X, y)
    return model

def generate_forecast(model, hourly_df, steps=24):
    forecast = []
    prev = hourly_df['Global_active_power'].iloc[-1]
    last_time = hourly_df.index[-1]
    for i in range(steps):
        ts = last_time + pd.Timedelta(hours=i+1)
        X_pred = pd.DataFrame({
            'Global_active_power': [prev],
            'hour': [ts.hour],
            'dayofweek': [ts.dayofweek],
            'month': [ts.month]
        })
        pred = model.predict(X_pred)[0]
        forecast.append((ts, pred))
        prev = pred
    return pd.Series([p for _, p in forecast], index=[t for t, _ in forecast])

@st.cache_resource
def train_anomaly_model(hourly_df):
    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(hourly_df[['Global_active_power']])
    return iso

# Train and Predict
forecast_model = train_forecast_model(hourly)
forecast_series = generate_forecast(forecast_model, hourly)
iso_model = train_anomaly_model(hourly)
hourly['anomaly'] = iso_model.predict(hourly[['Global_active_power']]) == -1

# Dashboard Navigation
st.title("🔌 WattsGuard AI Dashboard")
page = st.sidebar.radio("Navigate", [
    "Real-Time", "Daily Summary", "Forecast",
    "Profiling", "Weekly", "Anomalies",
    "Hourly Profile", "Goals"
])

# Render Screens
if page == "Real-Time":
    st.header("Function 1: Real-Time Monitoring")
    fig, ax = plt.subplots()
    ax.plot(hourly[-24:].index, hourly['Global_active_power'][-24:])
    ax.set_ylabel("kW")
    st.pyplot(fig)

elif page == "Daily Summary":
    st.header("Function 2: Daily Usage Summary")
    date = st.date_input("Select a date", df.index.date[-1])
    daily_data = hourly[hourly.index.date == date]
    fig, ax = plt.subplots()
    ax.plot(daily_data.index, daily_data['Global_active_power'], marker='o')
    ax.set_ylabel("kW")
    st.pyplot(fig)

elif page == "Forecast":
    st.header("Function 3: AI Forecast (Next 24h)")
    past = hourly['Global_active_power'][-48:]
    fig, ax = plt.subplots()
    ax.plot(past.index, past, label='Past')
    ax.plot(forecast_series.index, forecast_series, label='Forecast')
    ax.legend()
    ax.set_ylabel("kW")
    st.pyplot(fig)

elif page == "Profiling":
    st.header("Function 4: Appliance Profiling")
    fig, ax = plt.subplots()
    hourly[-24:][['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].plot(ax=ax)
    ax.set_ylabel("Watt-hours")
    st.pyplot(fig)

elif page == "Weekly":
    st.header("Function 5: Weekly Consumption")
    fig, ax = plt.subplots()
    daily[-7:]['Global_active_power'].plot(kind='bar', ax=ax)
    ax.set_ylabel("kWh")
    st.pyplot(fig)

elif page == "Anomalies":
    st.header("Function 6: Anomaly Detection")
    fig, ax = plt.subplots()
    ax.plot(hourly.index, hourly['Global_active_power'], label="Usage")
    ax.scatter(hourly[hourly['anomaly']].index,
               hourly[hourly['anomaly']]['Global_active_power'],
               color='red', s=15, label="Anomaly")
    ax.set_ylabel("kW")
    ax.legend()
    st.pyplot(fig)

elif page == "Hourly Profile":
    st.header("Function 7: Hourly Usage Profile")
    hourly_avg = df.groupby(df.index.hour)['Global_active_power'].mean()
    fig, ax = plt.subplots()
    hourly_avg.plot(kind='bar', ax=ax)
    ax.set_ylabel("kW")
    st.pyplot(fig)

elif page == "Goals":
    st.header("Function 8: Goals vs Actual")
    goal = st.number_input("Set your daily goal (kWh)", value=20.0)
    last_week = daily['Global_active_power'][-7:]
    fig, ax = plt.subplots()
    ax.plot(last_week.index, last_week.values, label="Actual")
    ax.plot(last_week.index, [goal]*7, linestyle='--', label="Goal")
    ax.legend()
    st.pyplot(fig)
