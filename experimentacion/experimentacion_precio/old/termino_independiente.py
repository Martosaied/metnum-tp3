from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import ModeloPrecio
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score, mean_absolute_error
from pathlib import Path
import json

df = pd.read_csv('../data/train.csv')

modelos = {
    'v1': ModeloPrecio.ModeloPrecioV1,
    'v2': ModeloPrecio.ModeloPrecioV2,
    'm2': ModeloPrecio.ModeloPrecioMetrosCuadrados,
    'm2_segmentado': ModeloPrecio.ModeloPrecioMetrosCuadradosSeg,
}

KFOLD_K = 1
accuracies_by_split = 0

columnas_piolas = ['centroscomercialescercanos','escuelascercanas','piscina','usosmultiples','banos', 'habitaciones', 'antiguedad', 'metroscubiertos', 'metrostotales', 'garages']
rms = []
rmsle = []
mae = []
r2 = []
    
mp = modelos['v2'](df, columnas_piolas)
df_predicted = mp.run(df, ['tipodepropiedad','provincia'])

df_predicted['precio'] = df_predicted['precio'].copy().apply(lambda x: x if x > 0 else 0) 
rms = mean_squared_error(df["precio"], df_predicted["precio"], squared=False)
rmsle = mean_squared_log_error(df["precio"],  df_predicted['precio'])
r2 = r2_score(df["precio"], df_predicted["precio"])
mae = mean_absolute_error(df["precio"], df_predicted["precio"])


print([rms, rmsle, r2, mae])
