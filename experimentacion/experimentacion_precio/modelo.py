import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import ModeloPrecio
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import metnum

df = pd.read_csv('../../data/train.csv')

modelos = {
    'v1': ModeloPrecio.ModeloPrecioV1,
    'm2': ModeloPrecio.ModeloPrecioMetrosCuadrados,
    'm2_segmentado': ModeloPrecio.ModeloPrecioMetrosCuadradosSeg,
}

mp = modelos['m2_segmentado'](df)
df_predicted = mp.run(df, ['tipodepropiedad','idzona' ,'antiguedad'])

print(df_predicted["precio"])
print(mp.get_df()["precio"])

rms = mean_squared_error(mp.get_df()["precio"], df_predicted["precio"], squared=False)
print(rms)

df_predicted['precio'] = df_predicted['precio'].apply(lambda x: x if x > 0 else 0) 
rmsle = mean_squared_log_error(mp.get_df()["precio"], df_predicted["precio"])
print(rmsle) 
