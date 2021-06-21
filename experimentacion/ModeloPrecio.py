from typing import Dict
import pandas as pd
import numpy as np

from pandas.core.frame import DataFrame
# from pandas.core.frame import DataFrame, DataFrameGroupBy
import metnum

class ModeloPrecio:
    def __init__(self, df):
        self.df: DataFrame = df
        self.grupos: Dict[str, metnum.LinearRegression] = {}
        self.linear_regressor_segmentos: Dict[str,
                                              metnum.LinearRegression] = {}

        self.columnas_piolas = ['banos', 'habitaciones', 'antiguedad',
                   'metroscubiertos', 'metrostotales', 'garages']


        self.categorias_chotas = {}
        self.modelo_fallback = None

    def clean(self, df, dropna=True):
        
        for columna in self.columnas_piolas:
            mean = df[columna].mean()
            mean = mean if not np.isnan(mean) else 0
            df[columna].fillna(mean, inplace=True)

        if dropna:
            df.dropna(inplace=True)

        df.drop(['id','titulo', 'descripcion', 'fecha', 'tipodepropiedad', 'direccion',
                   'ciudad', 'provincia', 'idzona', 'lat', 'lng'], axis=1, inplace=True)

    def segmentar(self, segmentos):
        # idea: dividir la ciudad en norte y sur
        # idea: en vez de segmentar a priori con categorias fijas, usar las propiedades mas cercanas en cuando a lat y lng para entrenar el modelo
        # lat y lng son datos que solo tiene la mitad del dataset. podemos quizas tomarlo en cuenta en el modelo con un peso que haga que influya pero poco
        # agrupar los tipos de propiedades similares que tienen pocas instancias??

        self.segmentos = segmentos
        dfGroupBy = self.df.groupby(segmentos, dropna=False)
        dfGroupByCiudad = self.df.groupby(segmentos[1], dropna=False)
        dfGroupByTipop = self.df.groupby(segmentos[0], dropna=False)

        for name, group in dfGroupBy:
            (tipop, ciudad) = name
            tipop = str(tipop)
            ciudad = str(ciudad)
            groupToSet = None
            if tipop == 'nan' and ciudad == 'nan':
                groupToSet = self.df.copy() # corregir esto
            elif tipop == 'nan':
                groupToSet = dfGroupByCiudad.get_group(ciudad)
            elif ciudad == 'nan':
                groupToSet = dfGroupByTipop.get_group(tipop)
            else:
                groupToSet = group

            if not tipop in self.grupos:
                self.grupos[tipop] = { ciudad: self.fit(groupToSet) }
            else:
                self.grupos[tipop][ciudad] = self.fit(groupToSet)

    def feature_engeneering(self):
        self.df['buena zona'] = (self.df['centroscomercialescercanos'] > 0) & (
            self.df['escuelascercanas'] > 0) & ('hospital' in self.df['descripcion'])
        self.df['seguro'] = ('vigilancia' in self.df['descripcion']) | (
            'seguridad' in self.df['descripcion'])
        # hacer feature engeneering para poder darle lat y lng a las props que no tienen

    def fit(self, group):
        self.clean(group)
        sinPrecio = group.drop(['precio'], axis=1).values
        precios = group['precio'].values.reshape(-1, 1)
        linear_regressor = metnum.LinearRegression()
        linear_regressor.fit(sinPrecio, precios)
        return linear_regressor

    def predict(self, dfToPred: DataFrame):
        segmentedPred = dfToPred.groupby(self.segmentos, dropna=False)
        result = []
        for name, group in segmentedPred:
            (tipop, ciudad) = name
            tipop = str(tipop)
            ciudad = str(ciudad)
            # self.feature_engeneering(group)
            self.clean(group, False)
            group["precio"] = self.grupos[tipop][ciudad].predict(group.values)

            result.append(group)

        return pd.concat(result).sort_index()

