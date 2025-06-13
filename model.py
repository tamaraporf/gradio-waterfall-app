from pycaret.regression import *
import shap
import pandas as pd


CLASS_TO_CODE = {
    'ExtraTreesRegressor': 'et',
    'RandomForestRegressor': 'rf',
    'XGBRegressor': 'xgboost',
    'LGBMRegressor': 'lightgbm',
    'CatBoostRegressor': 'catboost',
    'DecisionTreeRegressor': 'dt',
    'LinearRegression': 'lr',
}

def get_model(df_filtered):
    df_filtered_aux = df_filtered['TIPO']
    df_filtered = df_filtered.drop(['TIPO'], axis=1)
    setup(df_filtered, target='CPMAT', session_id=42, verbose=False)
    model_obj = compare_models()
    model_class = type(model_obj).__name__
    model_code = CLASS_TO_CODE.get(model_class)
    print(f"\n Melhor modelo: {model_class}")
    print(f"\n Código do modelo: {model_code}")
    if not model_code:
        print("Modelo vencedor não encontrado na lista. Retornando o modelo treinado diretamente.")
        return model_obj

    model_winner = create_model(model_code)
    return model_winner, df_filtered_aux


def visualize_model_performance(model_winner):
    plt.rcParams["font.family"] = "DejaVu Sans"
    print("\n\t-> Plotando importância das features:")
    plot_model(model_winner, plot='feature')

    print("\n\t -> Interpretando modelo com SHAP:")
    interpret_model(model_winner)
    return model_winner


def calculate_shap_values(model_winner):
    # dados de treino
    X_train = get_config('X_train')
    y_train = get_config('y_train')

    # dados de teste
    X_test = get_config('X_test')
    y_test = get_config('y_test')

    # dados transformados de teste, usados para treinar
    X_test_transformed = get_config("X_test_transformed")

    # criando o explicador shap
    explainer = shap.Explainer(model_winner, X_test_transformed)

    # gerando os valores shap a para os dados transformados
    shap_values = explainer(X_test_transformed)

    # Prepare a tabela para armazenar os valores
    shap_table = []

    # Itera por cada linha no X_test_transformed
    for i in range(len(X_test_transformed)):
        # Para cada registro, associe os valores SHAP com os valores da feature original
        row_data = {
            "base_value": shap_values.base_values[i],  # Valor base
            "predicted_value": shap_values.values[i].sum() + shap_values.base_values[i],  # Predição via SHAP
        }

        # Adiciona os valores SHAP e as variáveis originais
        for j, feature_name in enumerate(X_test_transformed.columns):
            row_data[f"{feature_name}_shap"] = shap_values[i].values[j]

        shap_table.append(row_data)

    # Crie o DataFrame com os valores
    df_shap = pd.DataFrame(shap_table, index=X_test.index)

    return X_test, y_test, X_test_transformed, df_shap