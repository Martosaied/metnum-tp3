from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pandas.core.frame import DataFrame
import metnum
import math


class ModeloBanoAbstract:
    def __init__(self, df, picked_columns, is_scikit=False):
        self.df: DataFrame = df
        self.picked_columns = picked_columns
        self.df_predict: DataFrame = None
        self.linear_regressor = metnum.LinearRegression if not is_scikit else LinearRegression

    def run(self, df_predict) -> DataFrame:
        """Ejecuta el modelo instanciado que implementa esta clase abstracta"""
        pass

    def get_df(self):
        return self.df


class ModeloBano(ModeloBanoAbstract):

    def run(self, df_predict) -> DataFrame:
        self.df_predict = df_predict
        self.clean()
        self.fit()
        return self.predict()

    def clean(self):
        self.df = self.df[self.picked_columns + ['banos']]
        self.df_predict = self.df_predict[self.picked_columns]
        for columna in self.picked_columns:
            mean = round(self.df[columna].mean())
            mean_pr = round(self.df_predict[columna].mean())
            print(mean)
            mean = mean if not math.isnan(mean) else 0
            self.df[columna].fillna(mean, inplace=True)
            self.df_predict[columna].fillna(mean_pr, inplace=True)

    def fit(self):
        X = self.df.drop('banos', axis=1).values
        y = self.df['banos'].values
        y = y.reshape(len(y), 1)
        self.linear_regressor_instance = self.linear_regressor()
        self.linear_regressor_instance.fit(X, y)

    def predict(self):
        response = self.df_predict.copy()
        response['banos'] = self.linear_regressor_instance.predict(
            response)
        return response

    def feature_engeneering(self):
        return 

    def get_test_df(self):
        return self.df_predict


class ModeloBanoSeg(ModeloBanoAbstract):
    def run(self, df_predict, segmentos) -> DataFrame:
        self.segmentos = segmentos
        self.df_predict = df_predict
        self.segmentar(segmentos)
        return self.predict()

    def segmentar(self, segmentos):
        self.df_grouped = self.df.groupby(segmentos, dropna=False)
        self.df_predict_grouped = self.df_predict.groupby(segmentos, dropna=False)
        
    def predict(self):
        result = []
        for name, group in self.df_predict_grouped:
            fit_df = self.get_segment(name)

            mp = ModeloBano(fit_df, self.picked_columns)
            df_predicted = mp.run(group)

            result.append(df_predicted)
        
        self.df_predict = pd.concat(result).sort_index() 
        return self.df_predict['banos'].apply(round)
    
    def get_segment(self, name):
        segment_values = np.asarray(name)
        try:
            if not 'nan' in segment_values:
                return self.df_grouped.get_group(name)

            conditions = ''
            for index, segmento in enumerate(self.segmentos):
                if not str(segment_values[index]) == 'nan':
                    conditions += f"{segmento} == '{segment_values[index]}' & "
            
            return self.df[self.df.eval(conditions[:-2])]
        except:
            return self.df


    def get_df(self):
        return self.df.loc[self.df_predict.index]
