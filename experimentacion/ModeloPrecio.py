from typing import Dict
import pandas as pd
from pandas.core.frame import DataFrame, DataFrameGroupBy
from pandas.core.series import Series
import metnum




class ModeloPrecio:
    def __init__(self, df):
        self.df: DataFrame = df
        self.grouped: DataFrameGroupBy = None
        self.linear_regressor_segmentos: Dict[str,
                                              metnum.LinearRegression] = {}

        self.columnas_piolas = ['banos', 'habitaciones', 'antiguedad',
                   'metroscubiertos', 'metrostotales', 'garages']

    def clean(self, df: DataFrame):
        
        for columna in self.columnas_piolas:
            df[columna].fillna(df[columna].mean(), inplace=True)

        df.dropna(inplace=True)

        df.drop(['id', 'titulo', 'descripcion', 'fecha', 'tipodepropiedad', 'direccion',
                   'ciudad', 'provincia', 'idzona', 'lat', 'lng'], axis=1, inplace=True)

    def segmentar(self, segmentos) -> DataFrameGroupBy:
        # idea: dividir la ciudad en norte y sur
        # idea: en vez de segmentar a priori con categorias fijas, usar las propiedades mas cercanas en cuando a lat y lng para entrenar el modelo
        # lat y lng son datos que solo tiene la mitad del dataset. podemos quizas tomarlo en cuenta en el modelo con un peso que haga que influya pero poco
        # agrupar los tipos de propiedades similares que tienen pocas instancias??
        self.segmentos = segmentos
        self.grouped = self.df.groupby(segmentos)

    def feature_engeneering(self, df: DataFrame):
        df['buena zona'] = (self.df['centroscomercialescercanos'] > 0) & (
            self.df['escuelascercanas'] > 0) & ('hospital' in self.df['descripcion'])
        df['seguro'] = ('vigilancia' in self.df['descripcion']) | (
            'seguridad' in self.df['descripcion'])
        # hacer feature engeneering para poder darle lat y lng a las props que no tienen

    def fit(self):
        for gName, group in self.grouped:
            self.clean(group)
            sinPrecio = group.drop('precio').values
            precios = group['precios'].values.reshape(-1, 1)
            self.linear_regressor_segmentos[gName] = metnum.LinearRegression()
            self.linear_regressor_segmentos[gName].fit(sinPrecio, precios)

    def predict(self, dfToPred: DataFrame):
        segmentedPred = dfToPred.groupby(self.segmentos)

        for gName, group in segmentedPred:
            self.feature_engeneering(group)
            self.clean(group)
            self.linear_regressor_segmentos[gName].predict(group.values)
