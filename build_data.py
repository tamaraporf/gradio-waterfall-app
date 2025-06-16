import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    print('Carregando dados')
    return df

def get_data(df, col_interest):
    # df = df.toPandas()

    # Apenas onde col_interest > 0
    df_filtered = df[df[col_interest] > 0].copy()

    # Salva a coluna original TIPO
    df_aux = df_filtered[['TIPO']].copy()

    # Mapeia o grupo da campanha e agrupa
    def map_grupo(tipo):
        if tipo in ['MARCA', 'OPORTUNIDADES', 'GENERICA', 'CURSOS']:
            return 'SEARCH'
        elif tipo in ['ASC', 'RMKT', 'LAL']:
            return 'META'
        elif tipo in ['PMAX']:
            return 'PMAX'
        elif tipo in ['DEMAND GEN']:
            return 'DEMAND GEN'
        else:
            return 'OUTROS'

    df_aux['TIPO_CAMPANHA'] = df_aux['TIPO'].apply(map_grupo)

    # Remove colunas não usadas no modelo
    df_filtered = df_filtered.drop(['DATA', 'GRUPO'], axis=1)
    print('Filtra os dados e salva o df_filtered e df_aux')

    return df_filtered, df_aux


def transformer_data(df):
    # filtra as colunas
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('CPMAT')

    # Label encoding só nas categóricas
    le = LabelEncoder()
    df[categorical_cols] = df[categorical_cols].apply(LabelEncoder().fit_transform)

    # Normalização apenas nas contínuas
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print('Transformando dados')

    return df