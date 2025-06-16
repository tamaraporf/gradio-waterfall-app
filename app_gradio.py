import os
import pandas as pd
import gradio as gr

from build_data import load_data, get_data, transformer_data
from model import get_model, calculate_shap_values
from visualizations import plot_shap_waterfall_percentual_by_campanha2

SHAP_FILE = "shap_values.parquet"


def preparar_dados(producao=True):
    if producao and os.path.exists(SHAP_FILE):
        print("✔️ Carregando SHAP values do arquivo salvo.")
        df_shap_values = pd.read_parquet(SHAP_FILE)
    else:
        print("⚙️ Treinando modelo e calculando SHAP values.")
        df_loaded = load_data("df_pandas.csv")
        df_filtered, df_aux = get_data(df_loaded, "CPMAT")
        df_code = transformer_data(df_filtered)

        model_winner, df_filtered_aux = get_model(df_code)
        X_test, y_test, X_test_transformed, df_shap_values = calculate_shap_values(
            model_winner
        )

        df_shap_values = pd.concat(
            [df_shap_values, df_aux[["TIPO_CAMPANHA", "DATA"]]], axis=1
        )
        df_shap_values.dropna(inplace=True)

        df_shap_values.to_parquet(SHAP_FILE, index=False)

    return df_shap_values


# def iniciar_app(producao=True):
#     df_shap_values = preparar_dados(producao)
#     campanhas = sorted(df_shap_values["TIPO_CAMPANHA"].dropna().unique())
#
#     def gerar_grafico(campanha_escolhida, limite=150):
#         try:
#             image = plot_shap_waterfall_percentual_by_campanha2(
#                 df_shap_values, campanha_escolhida, limite
#             )
#             return image
#         except Exception as e:
#             return f"Erro ao gerar gráfico: {str(e)}"
#
#     with gr.Blocks() as demo:
#         gr.Markdown("## MARTECH - Waterfall por Campanha")
#
#         with gr.Row():
#             dropdown = gr.Dropdown(choices=campanhas, label="Escolha a campanha")
#             slider = gr.Slider(
#                 minimum=0, maximum=500, value=150, label="Limite mínimo SHAP (absoluto)"
#             )
#
#         output = gr.Image(type="pil")
#         btn = gr.Button("Gerar Gráfico")
#
#         btn.click(fn=gerar_grafico, inputs=[dropdown, slider], outputs=output)
#
#     return demo
#
#
# demo = iniciar_app()
# demo.launch()

def iniciar_app(producao=True):
    df_shap_values = preparar_dados(producao)
    campanhas = sorted(df_shap_values["TIPO_CAMPANHA"].dropna().unique())
    datas = sorted(df_shap_values["DATA"].dropna().unique())

    def gerar_grafico(campanha_escolhida, data_escolhida, limite=150):
        try:
            image, _ = plot_shap_waterfall_percentual_by_campanha2(
                df_shap_values, campanha_escolhida, data_escolhida, limite
            )
            return image
        except Exception as e:
            return f"Erro ao gerar gráfico: {str(e)}"

    with gr.Blocks() as demo:
        gr.Markdown("## MARTECH - Waterfall por Campanha")

        with gr.Row():
            dropdown_campanha = gr.Dropdown(choices=campanhas, label="Escolha a campanha")
            dropdown_data = gr.Dropdown(choices=datas, label="Escolha a data")
            slider = gr.Slider(
                minimum=0, maximum=500, value=150, label="Limite mínimo SHAP (absoluto)"
            )

        output = gr.Image(type="pil")
        btn = gr.Button("Gerar Gráfico")

        btn.click(
            fn=gerar_grafico,
            inputs=[dropdown_campanha, dropdown_data, slider],
            outputs=output
        )

    return demo


demo = iniciar_app()
demo.launch(share=True, show_error=True)