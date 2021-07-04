from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import ModeloBano
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score, mean_absolute_error
from pathlib import Path

df = pd.read_csv('../../data/train.csv')
df = df[df['banos'].notna()]
df.reset_index(drop=True, inplace=True)

modelos = {
    'm': ModeloBano.ModeloBano,
    'm_segmentado': ModeloBano.ModeloBanoSeg,
    'v2': ModeloBano.ModeloV2
}

KFOLD_K = 5
kf = KFold(n_splits=KFOLD_K, shuffle=True)
columnas_piolas = ["precio","habitaciones","metroscubiertos", "metrostotales"]

segmentaciones = [['tipodepropiedad','ciudad'], ['tipodepropiedad'], ['tipodepropiedad', 'provincia'], ['ciudad']]

result = {}
for segmentacion in segmentaciones:
    rms = []
    rmsle = []
    mae = []
    r2 = []

    for train_index, test_index in kf.split(df):
        df_train, df_test = df.loc[train_index], df.loc[test_index]
        
        mp = modelos['v2'](df_train, columnas_piolas)
        df_predicted = mp.run(df_test, segmentacion)

        df_predicted['banos'] = df_predicted['banos'].copy().apply(lambda x: x if x > 0 else 0) 
        rms.append(mean_squared_error(df_test["banos"], df_predicted["banos"], squared=False))
        rmsle.append(mean_squared_log_error(df_test["banos"],  df_predicted['banos']))
        r2.append(r2_score(df_test["banos"], df_predicted["banos"]))
        mae.append(mean_absolute_error(df_test["banos"], df_predicted["banos"]))

    rms_result = np.sum(rms) / KFOLD_K 
    rmsle_result = np.sum(rmsle) / KFOLD_K 
    r2_result = np.sum(r2) / KFOLD_K 
    mae_result = np.sum(mae) / KFOLD_K

    result[', '.join(segmentacion)] = [rms_result, rmsle_result, r2_result, mae_result]


out = pd.DataFrame.from_dict(result, orient='index', columns=['RMSE', 'RMSLE', 'R^2', 'MAE']).to_latex(column_format='p{6cm}|r|r|r|r')
print(out)