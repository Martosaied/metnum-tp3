from typing import Dict, Tuple
from numpy import ndarray
import pandas as pd
from pandas.core.frame import DataFrame
import math
import metnum


class ModeloPrecio:
    def __init__(self, df):
        self.df: DataFrame = df
        self.grouped = None
        self.linear_regressor_segmentos: Dict[str,
                                              metnum.LinearRegression] = {}

        self.columnas_piolas = ['banos', 'habitaciones', 'antiguedad',
                                'metroscubiertos', 'metrostotales', 'garages']

    def clean(self, df: DataFrame, dropna=True):

        for columna in self.columnas_piolas:
            mean = df[columna].mean()
            mean = mean if not math.isnan(mean) else 0
            df[columna].fillna(mean, inplace=True)

        if dropna:
            df.dropna(inplace=True)

        df.drop(['id', 'titulo', 'descripcion', 'fecha', 'tipodepropiedad', 'direccion',
                 'ciudad', 'provincia', 'idzona', 'lat', 'lng'], axis=1, inplace=True)

        #negativos = df[df<0]
        #df.drop(df[negativos].index, axis=0, inplace=True)

    def segmentar(self, segmentos):
        # idea: dividir la ciudad en norte y sur
        # idea: en vez de segmentar a priori con categorias fijas, usar las propiedades mas cercanas en cuando a lat y lng para entrenar el modelo
        # lat y lng son datos que solo tiene la mitad del dataset. podemos quizas tomarlo en cuenta en el modelo con un peso que haga que influya pero poco
        # agrupar los tipos de propiedades similares que tienen pocas instancias??
        self.segmentos = segmentos
        self.grouped = self.df.groupby(segmentos, dropna=False)

    def feature_engeneering(self, df: DataFrame):
        df['buena zona'] = (self.df['centroscomercialescercanos'] > 0) & (
            self.df['escuelascercanas'] > 0) & ('hospital' in self.df['descripcion'])
        df['seguro'] = ('vigilancia' in self.df['descripcion']) | (
            'seguridad' in self.df['descripcion'])
        # hacer feature engeneering para poder darle lat y lng a las props que no tienen

    def fit(self):
        for gName, group in self.grouped:
            self.fit_model(gName, group)

        # precalculo el modelo que tiene a todas las instancias
        self.fit_model(math.nan, self.df.copy(deep=True))


    def fit_model(self, gName, group):
        self.feature_engeneering(group)
        self.clean(group)
        sinPrecio = group.drop('precio', axis=1).values
        precios = group['precio'].values.reshape(-1, 1)
        self.linear_regressor_segmentos[gName] = metnum.LinearRegression()
        self.linear_regressor_segmentos[gName].fit(sinPrecio, precios)

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
                    
                    for gName2, group2 in newGrouped:  # esto es fittear adentro de predict, sorry not sorry

                        gName2 = (gName2,) if type(gName2) is str else gName2 # (que vergaaaa)
                        self.fit_model(gName2, group2)

                gName = nameNoNAN

            self.feature_engeneering(group)
            self.clean(group, False)
            gName = gName if gName in self.linear_regressor_segmentos else math.nan
            group["precio"] = self.linear_regressor_segmentos[gName].predict(group.values)
            result.append(group)

        return pd.concat(result).sort_index()

    def contieneNaN(self, tupla: Tuple):
        for elem in tupla:
            if (not type(elem) is str) and math.isnan(elem):
                return True
        return False
