# data.py
import numpy as np
import pandas as pd

# ── Load CSVs ─────────────────────────────────────────────────────
meteo_df = pd.read_csv('../Data/meteo_tetiaroa_resampled.csv')

meteo_df = meteo_df.dropna(subset=['precip', 'precip1', 'precip2', 'precip4', 'UM', 'TM', 'date', 'time'])
precip_data = meteo_df['precip'].values
precip1_data = meteo_df['precip1'].values
precip2_data = meteo_df['precip2'].values
precip4_data = meteo_df['precip4'].values
hum_data = meteo_df['UM'].values
temperature_data = meteo_df['TM'].values
dates_data = meteo_df['date'].values
dates_dt = pd.to_datetime(dates_data)
time_data = meteo_df['time'].values

max_rain    = precip_data.max()