import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import ModeloPrecio
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import metnum

df = pd.read_csv('./data/train.csv')


mp = ModeloPrecio.ModeloPrecio(df)
# mp.feature_engeneering()
mp.segmentar(['tipodepropiedad', 'provincia'])

# temporal
#dfToPred = df[df['tipodepropiedad'].notnull()]
dfToPred = df

df_predicted: DataFrame = mp.predict(dfToPred)
#df_predicted['precio'] = df_predicted['precio'].apply(lambda x: x if x < 10000000 else 10000000) 
rms = mean_squared_error(dfToPred["precio"], df_predicted["precio"], squared=False)
print(rms)

df_predicted['precio'] = df_predicted['precio'].apply(lambda x: x if x > 0 else 0) 
rmsle = mean_squared_log_error(dfToPred["precio"], df_predicted["precio"])
print(rmsle) 
