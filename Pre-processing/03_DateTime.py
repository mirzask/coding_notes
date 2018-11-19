import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/fivethirtyeight/data/master/us-weather-history/KIND.csv')

df.head()
df.info()

# Convert column to datetime column

df['date_dt'] = pd.to_datetime(df['date'])

df.info() # see how it is now a datetime64 datatype


# Extract month

df['month'] = df['date_dt'].apply(lambda row: row.month)


# Extract year

df['year'] = df['date_dt'].apply(lambda row: row.year)


# Extract week
df['week'] = df['date_dt'].apply(lambda row: row.week)

# Extract day of the week
df['day_of_week'] = df['date_dt'].apply(lambda row: row.dayofweek)

# Extract day of the year
df['day_of_year'] = df['date_dt'].apply(lambda row: row.dayofyear)


# Extract quarter
df['quarter'] = df['date_dt'].apply(lambda row: row.quarter)

# Beginning of quarter?
df['beg_qtr'] = df['date_dt'].apply(lambda row: row.is_quarter_start)

# End of quarter?
df['end_qtr'] = df['date_dt'].apply(lambda row: row.is_quarter_end)


# Is it beginning of month?
df['beg_month'] = df['date_dt'].apply(lambda row: row.is_month_start)

# End of month?
df['end_month'] = df['date_dt'].apply(lambda row: row.is_month_end)

