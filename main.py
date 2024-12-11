import common_daily_load_curves as cdlc

ELECTRICITY_CONSUMPTION_DATA = "dataset/electricity_consumption.parquet"
SOCIOECONOMIC_DATA = "dataset/socioeconomic.parquet"
WEATHER_DATA = "dataset/weather.parquet"

cdlc.run(ELECTRICITY_CONSUMPTION_DATA)
