import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
from fbprophet import Prophet

mpl.rcParams['figure.figsize']=(10,8)
mpl.rcParams['axes.grid'] = False

df = pd.read_csv('https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])   
df.info()
df =df.set_index('timestamp').resample('H').mean()

fig = px.line(df.reset_index(), x = 'timestamp', y='value', title='NYC Taxi Demand')
fig.update_xaxes(
    rangeslider_visible = True,
    rangeselector = dict(
        buttons=list([
            dict(count=1, label='1y',step='year',stepmode='backward'),
            dict(count=2, label='3y',step='year',stepmode='backward'),
            dict(count=3, label='5y',step='year',stepmode='backward'),
            dict(step='all')
            
            ])
        )
    )
fig.show()

taxi_df = df.reset_index()[['timestamp','value']].rename({'timestamp':'ds','value':'y'}, axis = 'columns')

train = taxi_df[(taxi_df['ds']>='2014-07-01') & (taxi_df['ds']<='2015-01-27')]
test = taxi_df[(taxi_df['ds']>'2015-01-27')]

m = Prophet(changepoint_range=0.95)
m.fit(train)
future = m.make_future_dataframe(periods=119, freq='H')
forecast = m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()

results = pd.concat([taxi_df.set_index('ds')['y'], forecast.set_index('ds')[['yhat','yhat_lower','yhat_upper']]], axis=1)
fig1 = m.plot(forecast)
comp = m.plot_components(forecast)

results['error'] = results['y'] - results['yhat']
results['uncertainty'] = results['yhat_upper'] - results['yhat_lower']
results[results['error'].abs()>1.5*results['uncertainty']]
results['anomaly'] = results.apply(lambda x: 'Yes' if(np.abs(x['error']) > 1.5*x['uncertainty']) else 'No', axis=1)


fig = px.line(results.reset_index(), x = 'ds', y='y', color = 'anomaly',title='NYC Taxi Demand')
fig.update_xaxes(
    rangeslider_visible = True,
    rangeselector = dict(
        buttons=list([
            dict(count=1, label='1y',step='year',stepmode='backward'),
            dict(count=2, label='3y',step='year',stepmode='backward'),
            dict(count=3, label='5y',step='year',stepmode='backward'),
            dict(step='all')
            
            ])
        )
    )
fig.show()

import fbprophet
# print version number
print('Prophet %s' % fbprophet.__version__)

















