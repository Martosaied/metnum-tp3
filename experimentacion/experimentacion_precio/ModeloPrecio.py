from typing import Dict
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pandas.core.frame import DataFrame
# from pandas.core.frame import DataFrame, DataFrameGroupBy
import metnum
import math


class ModeloPrecioAbstract:
    def __init__(self, df, is_scikit=False):
        self.df: DataFrame = df
        self.df_predict: DataFrame = None
        self.linear_regressor = metnum.LinearRegression() if not is_scikit else LinearRegression()

    def run(self, df_predict) -> DataFrame:
        """Ejecuta el modelo instanciado que implementa esta clase abstracta"""
        pass

    def get_df(self):
        return self.df


class ModeloPrecioV1(ModeloPrecioAbstract):
    def __init__(self, df, is_scikit=False):
        super(ModeloPrecioV1, self).__init__(df, is_scikit)
        self.grupos: Dict[str, metnum.LinearRegression] = {}
        self.linear_regressor_segmentos: Dict[str,
                                              metnum.LinearRegression] = {}

        self.columnas_piolas = [
            'banos', 'habitaciones', 'antiguedad', 'metroscubiertos',
            'metrostotales', 'garages'
        ]

    def run(self, df):
        self.df_predict = df

        self.df_predict = self.feature_engeneering(self.df_predict)
        self.df = self.feature_engeneering(self.df)

        self.df = self.segmentar(['tipodepropiedad', 'provincia'], self.df)
        self.df_predict = self.segmentar(['tipodepropiedad', 'provincia'],
                                         self.df_predict)

        df_predicted: DataFrame = self.predict()
        return df_predicted

    def clean(self, df, dropna=True):

        for columna in self.columnas_piolas:
            mean = df[columna].mean()
            mean = mean if not np.isnan(mean) else 0
            df[columna].fillna(mean, inplace=True)

        if dropna:
            df.dropna(inplace=True)

        df.drop([
            'id', 'titulo', 'descripcion', 'fecha', 'tipodepropiedad',
            'direccion', 'ciudad', 'provincia', 'idzona', 'lat', 'lng'
        ],
                axis=1,
                inplace=True)

    def segmentar(self, segmentos, df):
        # idea: dividir la ciudad en norte y sur
        # idea: en vez de segmentar a priori con categorias fijas, usar las propiedades mas cercanas en cuando a lat y lng para entrenar el modelo
        # lat y lng son datos que solo tiene la mitad del dataset. podemos quizas tomarlo en cuenta en el modelo con un peso que haga que influya pero poco
        # agrupar los tipos de propiedades similares que tienen pocas instancias??

        self.segmentos = segmentos
        dfGroupBy = df.groupby(segmentos, dropna=False)
        dfGroupByCiudad = df.groupby(segmentos[1], dropna=False)
        dfGroupByTipop = df.groupby(segmentos[0], dropna=False)

        for name, group in dfGroupBy:
            (tipop, ciudad) = name
            tipop = str(tipop)
            ciudad = str(ciudad)
            groupToSet = None
            if tipop == 'nan' and ciudad == 'nan':
                groupToSet = df.copy()  # corregir esto
            elif tipop == 'nan':
                groupToSet = dfGroupByCiudad.get_group(ciudad)
            elif ciudad == 'nan':
                groupToSet = dfGroupByTipop.get_group(tipop)
            else:
                groupToSet = group

            if not tipop in self.grupos:
                self.grupos[tipop] = {ciudad: self.fit(groupToSet)}
            else:
                self.grupos[tipop][ciudad] = self.fit(groupToSet)

        return dfGroupBy

    def feature_engeneering(self, df):
        df['buena zona'] = (df['centroscomercialescercanos'] > 0) & (
            df['escuelascercanas'] > 0) & ('hospital' in df['descripcion'])
        df['seguro'] = ('vigilancia' in df['descripcion']) | (
            'seguridad' in df['descripcion'])
        return df
        # hacer feature engeneering para poder darle lat y lng a las props que no tienen

    def fit(self, group):
        self.clean(group)
        sinPrecio = group.drop(['precio'], axis=1).values
        precios = group['precio'].values.reshape(-1, 1)
        linear_regressor = metnum.LinearRegression()
        linear_regressor.fit(sinPrecio, precios)
        return linear_regressor

    def predict(self):
        result = []
        for name, group in self.df_predict:
            (tipop, ciudad) = name
            tipop = str(tipop)
            ciudad = str(ciudad)
            self.clean(group, False)
            group["precio"] = self.grupos[tipop][ciudad].predict(group.values)

            result.append(group)

        return pd.concat(result).sort_index()

    def get_df(self):
        result = []
        for name, group in self.df_predict:
            result.append(group)

        return pd.concat(result).sort_index()

class ModeloPrecioMetrosCuadrados(ModeloPrecioAbstract):
    def run(self, df_predict) -> DataFrame:
        self.df_predict = df_predict
        self.clean()
        self.fit()
        return self.predict()

    def clean(self):
        self.df = self.df[['metroscubiertos', 'metrostotales','precio']]
        self.df = self.df.dropna()
        self.df_predict = self.df_predict[['metroscubiertos','metrostotales']]
        self.df_predict = self.df_predict.dropna()

    def fit(self):
        X = self.df.drop('precio', axis=1).values
        y = self.df['precio'].values
        y = y.reshape(len(y), 1)
        self.linear_regressor.fit(X, y)

    def predict(self):
        self.df_predict['precio'] = self.linear_regressor.predict(
            self.df_predict)
        return self.df_predict


class ModeloPrecioMetrosCuadradosSeg(ModeloPrecioAbstract):
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

            mp = ModeloPrecioMetrosCuadrados(fit_df)
            df_predicted = mp.run(group)

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

        
