# ---IMPORTAÇÕES---

import pandas as pd
import gradio as gr
from build_data import *
from model import *
from visualizations import plot_shap_waterfall_percentual_by_campanha2

# ---RODANDO O BUILD DATA---

df_loaded = load_data("df_pandas.csv")
df_filtered, df_aux = get_data(df_loaded, "CPMAT")
df_code = transformer_data(df_filtered)


# ---TREINANDO O MODELO---
model_winner, df_filtered_aux = get_model(df_code)
X_test, y_test, X_test_transformed, df_shap_values = calculate_shap_values(model_winner)


# --- PEGA OS SHAP VALUES ---
df_shap_values = pd.concat([df_shap_values, df_aux[["TIPO_CAMPANHA"]]], axis=1)
df_shap_values.dropna(inplace=True, axis=0)


# --- SALVA OS SHAP VALUES EM UM ARQUIVO PARQUET ---
df_shap_values.to_parquet("shap_values.parquet", index=False)


# --- AVALIA AS CAMPANHAS DISPONIVEIS ----
campanhas = sorted(df_shap_values["TIPO_CAMPANHA"].dropna().unique())


# --- FUNÇÃO PARA O APP GRADIO ---
def gerar_grafico(campanha_escolhida, limite=150):
    try:
        image = plot_shap_waterfall_percentual_by_campanha2(
            df_shap_values, campanha_escolhida, limite
        )
        return image
    except Exception as e:
        return f"Erro ao gerar gráfico: {str(e)}"


# --- INTERFACE GRADIO ---
with gr.Blocks() as demo:
    gr.Markdown("### SHAP Waterfall por Campanha")

    with gr.Row():
        dropdown = gr.Dropdown(choices=campanhas, label="Escolha a campanha")
        slider = gr.Slider(
            minimum=0, maximum=500, value=150, label="Limite mínimo SHAP (absoluto)"
        )

    output = gr.Image(type="pil")
    btn = gr.Button("Gerar Gráfico")

    btn.click(fn=gerar_grafico, inputs=[dropdown, slider], outputs=output)

if __name__ == "__main__":
    demo.launch(share=True)
