from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pandas.core.frame import DataFrame
import metnum
import math


class ModeloPrecioAbstract:
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

class SegmentadoV2(ModeloPrecioAbstract):
    def __init__(self, df, picked_columns):
        super(SegmentadoV2, self).__init__(df, picked_columns)
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

        df = df[self.columnas_piolas + ['precio']]
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
        self.feature_engeneering(group)
        self.clean(group)
        sinPrecio = np.c_[group[self.columnas_piolas].values, np.ones(group.values.shape[0])] if self.termino_independiente else group[self.columnas_piolas].values
        precios = group['precio'].values.reshape(-1, 1)
        self.linear_regressor_segmentos[gName] = self.linear_regressor()
        self.linear_regressor_segmentos[gName].fit(sinPrecio , precios)

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

            self.feature_engeneering(group)
            self.clean(group, False)


            gName = gName if gName in self.linear_regressor_segmentos else math.nan
            sinPrecio = np.c_[group[self.columnas_piolas].values, np.ones(group.values.shape[0])] if self.termino_independiente else group[self.columnas_piolas].values
            group["precio"] = self.linear_regressor_segmentos[gName].predict(sinPrecio)
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

class SinSegmentar(ModeloPrecioAbstract):

    def run(self, df_predict, feature_engineering=False) -> DataFrame:
        self.feature_engeneering_attrs = ['buena zona', 'seguro'] if feature_engineering else []
        self.df_predict = df_predict
        if feature_engineering: self.feature_engeneering()
        self.clean()
        self.fit()
        return self.predict()

    def clean(self):
        self.df = self.df[self.picked_columns + self.feature_engeneering_attrs + ['precio']]
        self.df_predict = self.df_predict[self.picked_columns + self.feature_engeneering_attrs]
        for columna in self.picked_columns:
            mean = self.df[columna].mean()
            mean_pr = self.df_predict[columna].mean()
            mean = mean if not math.isnan(mean) else 0
            self.df[columna].fillna(mean, inplace=True)
            self.df_predict[columna].fillna(mean_pr, inplace=True)

    def fit(self):
        X = self.df.drop('precio', axis=1).values
        y = self.df['precio'].values
        y = y.reshape(len(y), 1)
        self.linear_regressor_instance = self.linear_regressor()
        self.linear_regressor_instance.fit(X, y)

    def predict(self):
        response = self.df_predict.copy()
        response['precio'] = self.linear_regressor_instance.predict(
            response)
        return response

    def feature_engeneering(self):
        self.df['buena zona'] = (self.df['centroscomercialescercanos'] > 0) & (
            self.df['escuelascercanas'] > 0) & ('hospital' in self.df['descripcion'])
        self.df['buena zona'] = self.df['buena zona'].astype(int)
        self.df['seguro'] = ('vigilancia' in self.df['descripcion']) | (
            'seguridad' in self.df['descripcion'])
        self.df['seguro'] = self.df['seguro'].astype(int)
        self.df_predict['buena zona'] = (self.df_predict['centroscomercialescercanos'] > 0) & (
            self.df_predict['escuelascercanas'] > 0) & ('hospital' in self.df_predict['descripcion'])
        self.df_predict['buena zona'] = self.df_predict['buena zona'].astype(int)
        self.df_predict['seguro'] = ('vigilancia' in self.df_predict['descripcion']) | (
            'seguridad' in self.df_predict['descripcion'])
        self.df_predict['seguro'] = self.df_predict['seguro'].astype(int)

    def get_test_df(self):
        return self.df_predict


class Segmentado(ModeloPrecioAbstract):
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

            mp = SinSegmentar(fit_df, self.picked_columns)
            df_predicted = mp.run(group, feature_engineering=True)

            result.append(df_predicted)
        
        self.df_predict = pd.concat(result).sort_index() 
        return self.df_predict
    
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

class ModeloPrecioV2FeatEng(SegmentadoV2):
    def __init__(self, df, seguro_buenaZona):
        self.seguro, self.buenaZona = seguro_buenaZona
        piolas = ["escuelascercanas","piscina","usosmultiples","banos","habitaciones","metroscubiertos"]
        if self.seguro:
            piolas.append('seguro')
        if self.buenaZona:
            piolas.append('buena zona')
        super(ModeloPrecioV2FeatEng, self).__init__(df, piolas)


    def feature_engeneering(self, df: DataFrame):
        if self.buenaZona:
            df['buena zona'] = (self.df['centroscomercialescercanos'] > 0) & (
                self.df['escuelascercanas'] > 0) & ('hospital' in self.df['descripcion'])
        if self.seguro:
            df['seguro'] = ('vigilancia' in self.df['descripcion']) | (
                'seguridad' in self.df['descripcion'])
