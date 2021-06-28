from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import ModeloPrecio
from sklearn.metrics import mean_squared_error, mean_squared_log_error

df = pd.read_csv('../data/train.csv')

modelos = {
    'v1': ModeloPrecio.ModeloPrecioV1,
    'v2': ModeloPrecio.ModeloPrecioV2,
    'm2': ModeloPrecio.ModeloPrecioMetrosCuadrados,
    'm2_segmentado': ModeloPrecio.ModeloPrecioMetrosCuadradosSeg,
}

mp = modelos['m2'](df)
df_predicted = mp.run(df, ['tipodepropiedad','provincia'])


rms = mean_squared_error(mp.get_df()["precio"], df_predicted["precio"], squared=False)
print(rms)

df_predicted['precio'] = df_predicted['precio'].apply(lambda x: x if x > 0 else 0) 
rmsle = mean_squared_log_error(mp.get_df()["precio"], df_predicted["precio"])
print(rmsle) 


rms = []
rmsle = []
KFOLD_K = 5
kf = KFold(n_splits=KFOLD_K, shuffle=True)
accuracies_by_split = 0

for train_index, test_index in kf.split(df):
    df_train, df_test = df[train_index], df[test_index]
    
    mp = modelos['m2'](df_train)
    df_predicted = mp.run(df_test, ['tipodepropiedad','provincia'])

    rms.append(mean_squared_error(mp.get_df()["precio"], df_predicted["precio"], squared=False))
    df_predicted['precio'] = df_predicted['precio'].apply(lambda x: x if x > 0 else 0) 
    rmsle.append(mean_squared_log_error(mp.get_df()["precio"], df_predicted["precio"], squared=False))
