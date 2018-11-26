import pandas as pd

# Convert datetime columns on import using `parse_dates` in pd.read_csv
times = [f'time{i}' for i in range(1, 11)]

train_df = pd.read_csv('data/CatchMeIfYouCan_Alice/train_sessions.csv', index_col='session_id', parse_dates=times)

train_df[times].dtypes



# More fun w/ Dates and Times

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
df['beg_qtr_bool'] = df['date_dt'].apply(lambda row: row.is_quarter_start) # boolean output
df['beg_qtr'] = df['date_dt'].apply(lambda row: row.is_quarter_start).astype(int) # integer output
print(df[["date_dt", "beg_qtr_bool", "beg_qtr"]].head())


# End of quarter?
df['end_qtr_bool'] = df['date_dt'].apply(lambda row: row.is_quarter_end) # boolean output
df['end_qtr'] = df['date_dt'].apply(lambda row: row.is_quarter_end).astype(int) # integer output
print(df[["date_dt", "end_qtr_bool", "end_qtr"]].head())

# Is it beginning of month?
df['beg_month_bool'] = df['date_dt'].apply(lambda row: row.is_month_start) # boolean output
df['beg_month'] = df['date_dt'].apply(lambda row: row.is_month_start).astype(int) # integer output

# End of month?
df['end_month_bool'] = df['date_dt'].apply(lambda row: row.is_month_end) # boolean output
df['end_month'] = df['date_dt'].apply(lambda row: row.is_month_end).astype(int) # integer output




# Time of day

hour = df['time'].apply(lambda ts: ts.hour)
morning = ((hour >= 7) & (hour <= 11)).astype('int')
day = ((hour >= 12) & (hour <= 18)).astype('int')
evening = ((hour >= 19) & (hour <= 23)).astype('int')
night = ((hour >= 0) & (hour <= 6)).astype('int')
