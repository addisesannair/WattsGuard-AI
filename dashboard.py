# dashboard_ai.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import xgboost as xgb

# 1) DATA LOADING & CLEANING
@st.cache_data
def load_data():
    df = pd.read_csv('household_power_consumption.csv', sep=';', header=None)
    # split the single column into proper fields
    df = df[0].str.split(',', expand=True)
    df.columns = [
        'Date', 'Time',
        'Global_active_power', 'Global_reactive_power',
        'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    df = df[df['Date'] != 'Date']  # drop header-rows
    # parse datetime
    df['Datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce'
    )
    # convert measurement columns to numeric
    numeric_cols = [
        'Global_active_power', 'Global_reactive_power',
        'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    # drop old text columns & set index
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    df.set_index('Datetime', inplace=True)
    return df

df = load_data()
hourly = df.resample('H').mean(numeric_only=True)
daily  = df.resample('D').sum(numeric_only=True)

# 2) TRAIN AI MODELS
@st.cache_resource
def train_forecast_model(hourly_df):
    hr = hourly_df.copy()
    hr['hour'] = hr.index.hour
    hr['dayofweek'] = hr.index.dayofweek
    hr['month'] = hr.index.month
    hr['target'] = hr['Global_active_power'].shift(-1)
    hr.dropna(inplace=True)
    features = ['Global_active_power', 'hour', 'dayofweek', 'month']
    X = hr[features]; y = hr['target']
    split = int(len(hr) * 0.7)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, verbosity=0)
    model.fit(X.iloc[:split], y.iloc[:split])
    return model

def generate_forecast(model, hourly_df, steps=24):
    preds = []
    last_val = hourly_df['Global_active_power'].iloc[-1]
    last_time = hourly_df.index[-1]
    for i in range(steps):
        ts = last_time + pd.Timedelta(hours=i+1)
        feat = pd.DataFrame({
            'Global_active_power': [last_val],
            'hour': [ts.hour],
            'dayofweek': [ts.dayofweek],
            'month': [ts.month]
        })
        pred = model.predict(feat)[0]
        preds.append((ts, pred))
        last_val = pred
    return pd.Series(
        [p for _, p in preds],
        index=pd.DatetimeIndex([t for t, _ in preds])
    )

@st.cache_resource
def train_anomaly_model(hourly_df):
    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(hourly_df[['Global_active_power']])
    return iso

# build models
forecast_model = train_forecast_model(hourly)
forecast_series = generate_forecast(forecast_model, hourly)
iso_model = train_anomaly_model(hourly)
hourly['anomaly'] = iso_model.predict(hourly[['Global_active_power']]) == -1

# 3) STREAMLIT UI
st.title("🔌 WattsGuard AI Dashboard")

nav = st.sidebar.radio("Navigation", [
    "Real-Time", "Daily Summary", "Forecast",
    "Profiling", "Weekly", "Anomalies",
    "Hourly Profile", "Goals"
])

if nav == "Real-Time":
    st.header("Real-Time (Last 24h)")
    data = hourly['Global_active_power'][-24:]
    fig, ax = plt.subplots()
    ax.plot(data.index, data.values)
    ax.set_ylabel("kW")
    st.pyplot(fig)

elif nav == "Daily Summary":
    st.header("Daily Usage Summary")
    date = st.date_input("Select Date", df.index.date[-1])
    day = hourly[hourly.index.date == date]['Global_active_power']
    fig, ax = plt.subplots()
    ax.plot(day.index, day.values, marker='o')
    ax.set_ylabel("kW")
    st.pyplot(fig)

elif nav == "Forecast":
    st.header("AI Forecast (Next 24h)")
    past = hourly['Global_active_power'][-48:]
    fig, ax = plt.subplots()
    ax.plot(past.index, past.values, label='Past 48h')
    ax.plot(forecast_series.index, forecast_series.values, label='Forecast')
    ax.set_ylabel("kW")
    ax.legend()
    st.pyplot(fig)

elif nav == "Profiling":
    st.header("Appliance Profiling")
    data = hourly[['Sub_metering_1','Sub_metering_2','Sub_metering_3']][-24:]
    fig, ax = plt.subplots()
    for col in data.columns:
        ax.plot(data.index, data[col], label=col)
    ax.legend()
    st.pyplot(fig)

elif nav == "Weekly":
    st.header("Weekly Consumption")
    week = daily['Global_active_power'][-7:]
    fig, ax = plt.subplots()
    ax.bar(week.index, week.values)
    ax.set_ylabel("kWh")
    st.pyplot(fig)

elif nav == "Anomalies":
    st.header("Anomaly Detection")
    fig, ax = plt.subplots()
    ax.plot(hourly.index, hourly['Global_active_power'], label='Usage')
    ax.scatter(
        hourly.index[hourly['anomaly']],
        hourly['Global_active_power'][hourly['anomaly']],
        color='red', s=10, label='Anomaly'
    )
    ax.legend()
    st.pyplot(fig)

elif nav == "Hourly Profile":
    st.header("Hourly Usage Profile")
    avg = df.groupby(df.index.hour)['Global_active_power'].mean()
    fig, ax = plt.subplots()
    ax.bar(avg.index, avg.values)
    ax.set_xlabel("Hour")
    ax.set_ylabel("kW")
    st.pyplot(fig)

elif nav == "Goals":
    st.header("Goals vs Actual")
    goal = st.number_input("Daily kWh Goal", value=20.0)
    last_week = daily['Global_active_power'][-7:]
    fig, ax = plt.subplots()
    ax.plot(last_week.index, last_week.values, label='Actual')
    ax.plot(last_week.index, [goal]*7, '--', label='Goal')
    ax.legend()
    st.pyplot(fig)
