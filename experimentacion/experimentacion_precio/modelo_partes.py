from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import ModeloPrecio
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from pathlib import Path
import json

df = pd.read_csv('../../data/train.csv')

modelos = {
    'v1': ModeloPrecio.ModeloPrecioV1,
    'v2': ModeloPrecio.ModeloPrecioV2,
    'm2': ModeloPrecio.ModeloPrecioMetrosCuadrados,
    'm2_segmentado': ModeloPrecio.ModeloPrecioMetrosCuadradosSeg,
}

KFOLD_K = 5
kf = KFold(n_splits=KFOLD_K, shuffle=True)
accuracies_by_split = 0

columnas_piolas = [            'banos', 'habitaciones', 'antiguedad', 'metroscubiertos',
            'metrostotales', 'garages']

def powerset(s):
    conjunto_de_partes = []
    x = len(s)
    for i in range(1 << x):
        conjunto_de_partes.append([s[j] for j in range(x) if (i & (1 << j))])
    return conjunto_de_partes

result = {}
print(powerset(columnas_piolas))
for picked_columns in powerset(columnas_piolas):
    rms = []
    rmsle = []
    if picked_columns == []:
        continue

    for train_index, test_index in kf.split(df):
        df_train, df_test = df.loc[train_index], df.loc[test_index]
        
        mp = modelos['m2'](df_train, picked_columns)
        df_predicted = mp.run(df_test)#, ['tipodepropiedad','provincia'])
        
        # HACK
        df_test_copy = df_test[picked_columns + ['precio']]
        df_test_copy = df_test_copy.dropna()

        
        rms.append(mean_squared_error(df_test_copy["precio"], df_predicted["precio"], squared=False))
        df_predicted['precio'] = df_predicted['precio'].apply(lambda x: x if x > 0 else 0) 
        rmsle.append(mean_squared_log_error(df_test_copy["precio"], df_predicted["precio"]))

    rms_result = np.sum(rms) / KFOLD_K 
    rmsle_result = np.sum(rmsle) / KFOLD_K 

    result['-'.join(picked_columns)] = np.array([rms_result, rmsle_result])

print(result)
print(json.dumps(result, indent = 4))



to_save = pd.DataFrame.from_dict(result)

myfile = Path('rms_rmsle_by_model/rms_rmsle_kfold_v2.npy')
myfile.touch(exist_ok=True)
with open('rms_rmsle_by_model/rms_rmsle_kfold_v2.npy', 'wb') as f:
    np.save(f, to_save)

errores = np.load('rms_rmsle_by_model/rms_rmsle_kfold_v2.npy')
# print(errores)