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
        self.df_predict = self.df_predict[self.picked_columns + ['banos']]
        for columna in self.picked_columns + ['banos']:
            self.df[columna].fillna(2, inplace=True)
            if not columna == 'banos':
                self.df_predict[columna].fillna(2, inplace=True)

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
        return self.df_predict['banos'].fillna(2).apply(round)
    
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


class ModeloV2(ModeloBanoAbstract):
    def __init__(self, df, picked_columns):
        super(ModeloV2, self).__init__(df, picked_columns)
        self.clean_df = self.df.copy()
        self.grouped = None
        self.linear_regressor_segmentos: Dict[str,
                                              metnum.LinearRegression] = {}

        self.columnas_piolas = picked_columns
        self.termino_independiente = False
    
    def run(self, df_predict, segmentos):
        self.segmentar(segmentos)
        self.fit()
        return self.predict(df_predict)

    def clean(self, df: DataFrame, dropna=True):

        for columna in self.columnas_piolas:
            mean = df[columna].mean()
            mean = mean if not math.isnan(mean) else 0
            df[columna].fillna(mean, inplace=True)

        df = df[self.columnas_piolas + ['banos']]
        if dropna:
            df.dropna(inplace=True)


    def segmentar(self, segmentos):
        self.segmentos = segmentos
        self.grouped = self.df.groupby(segmentos, dropna=False)

    def feature_engeneering(self, df: DataFrame):
        return

    def fit(self):
        for gName, group in self.grouped:
            self.fit_model(gName, group)

        # precalculo el modelo que tiene a todas las instancias
        self.fit_model(math.nan, self.df.copy(deep=True))


    def fit_model(self, gName, group):
        self.clean(group)
        banos = np.c_[group[self.columnas_piolas].values, np.ones(group.values.shape[0])] if self.termino_independiente else group[self.columnas_piolas].values
        banoss = group['banos'].values.reshape(-1, 1)
        self.linear_regressor_segmentos[gName] = self.linear_regressor()
        self.linear_regressor_segmentos[gName].fit(banos , banoss)

    def predict(self, dfToPred: DataFrame):
        segmentedPred = dfToPred.groupby(self.segmentos, dropna=False)
        result = []
        for gName, group in segmentedPred:
            if type(gName) is tuple and self.contieneNaN(gName):

                nameNoNAN = tuple([x for x in gName if type(x) is str])

                nameNoNAN = math.nan if len(nameNoNAN) == 0 else nameNoNAN

                if not nameNoNAN in self.linear_regressor_segmentos:
                    categoriasNoNaN = [self.segmentos[i] for i, x in enumerate(gName) if type(x) is str]
                    newGrouped = self.df.groupby(categoriasNoNaN)
                    
                    for gName2, group2 in newGrouped:

                        gName2 = (gName2,) if type(gName2) is str else gName2
                        self.fit_model(gName2, group2)

                gName = nameNoNAN

            self.clean(group, False)


            gName = gName if gName in self.linear_regressor_segmentos else math.nan
            banos = np.c_[group[self.columnas_piolas].values, np.ones(group.values.shape[0])] if self.termino_independiente else group[self.columnas_piolas].values
            group["banos"] = self.linear_regressor_segmentos[gName].predict(banos)
            result.append(group)

        result = pd.concat(result).sort_index()
        return result

    def contieneNaN(self, tupla: Tuple):
        for elem in tupla:
            if (not type(elem) is str) and math.isnan(elem):
                return True
        return False

    def get_df(self):
        return self.clean_df