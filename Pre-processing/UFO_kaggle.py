# Import

import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

ufo = pd.read_csv('../coding_notes/data/UFO/scrubbed.csv')

# EDA

ufo.head()
ufo.shape
ufo.columns
ufo.dtypes
ufo.info()

ufo['duration (seconds)'].head(10)
ufo['datetime'].head(10)

pd.to_datetime("10/10/1949 20:30", format='%m/%d/%Y %H:%M')

pd['time'] = pd.to_datetime(ufo['datetime'], format='%m/%d/%Y %H:%M')