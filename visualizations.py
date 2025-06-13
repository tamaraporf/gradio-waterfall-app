import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image
plt.rcParams["font.family"] = "DejaVu Sans"

def plot_shap_waterfall_by_campanha(df_shap_values, campanha_escolhida, limite=150):
    # Remove colunas duplicadas, se existirem
    df_shap_values = df_shap_values.loc[:, ~df_shap_values.columns.duplicated()]

    # Filtra a campanha desejada
    df_filtrado = df_shap_values[df_shap_values['TIPO_CAMPANHA'] == campanha_escolhida]

    # Seleciona apenas as colunas SHAP
    shap_cols = [col for col in df_filtrado.columns if col.endswith('_shap')]

    # Calcula a média dos SHAP values
    mean_shap = df_filtrado[shap_cols].mean()

    # Filtra somente contribuições relevantes (fora do intervalo ±limite)
    mean_shap = mean_shap[(mean_shap > limite) | (mean_shap < -limite)]
    mean_shap = mean_shap.sort_values(key=abs, ascending=False)

    # Valores e rótulos
    labels = mean_shap.index.tolist()
    values = mean_shap.values

    # Valor base e predição média
    base_value = df_filtrado['base_value'].mean()
    prediction = base_value + values.sum()

    # Cores
    colors = ['green' if v > 0 else 'red' for v in values]
    colors.insert(0, 'gray')
    labels.insert(0, 'Base value')
    values = np.insert(values, 0, base_value)

    # Base acumulada
    starts = np.cumsum([0] + list(values[:-1]))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(len(values)):
        ax.bar(labels[i], values[i], bottom=starts[i], color=colors[i])
        y = starts[i] + values[i]
        va = 'bottom' if values[i] > 0 else 'top'
        offset = 1.5 if values[i] > 0 else -1.5
        ax.text(i, y + offset * 0.01, f"{values[i]:+.2f}", ha='center', va=va, fontsize=8)

    ax.axhline(y=prediction, linestyle='--', color='black', label=f'Predição média: {prediction:.2f}')
    ax.set_ylabel('Contribuição média SHAP')
    ax.set_title(f'SHAP Waterfall - Campanha: {campanha_escolhida}')
    ax.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_shap_waterfall_percentual_by_campanha2(df_shap_values, campanha_escolhida, limite=150):
    df_shap_values = df_shap_values.loc[:, ~df_shap_values.columns.duplicated()]
    df_filtrado = df_shap_values[df_shap_values['TIPO_CAMPANHA'] == campanha_escolhida]
    shap_cols = [col for col in df_filtrado.columns if col.endswith('_shap')]
    mean_shap = df_filtrado[shap_cols].mean()
    base_value = df_filtrado['base_value'].mean()
    prediction = base_value + mean_shap.sum()
    mean_shap = mean_shap[(mean_shap > limite) | (mean_shap < -limite)]
    mean_shap_pct = (mean_shap / prediction) * 100
    base_value_pct = (base_value / prediction) * 100
    prediction_pct = 100
    mean_shap_pct = mean_shap_pct.sort_values(key=abs, ascending=False)

    labels = mean_shap_pct.index.tolist()
    values = mean_shap_pct.values
    colors = ['green' if v > 0 else 'red' for v in values]
    colors.insert(0, 'gray')
    labels.insert(0, 'Base value')
    values = np.insert(values, 0, base_value_pct)
    starts = np.cumsum([0] + list(values[:-1]))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(values)):
        ax.bar(labels[i], values[i], bottom=starts[i], color=colors[i], zorder=3)
        y = starts[i] + values[i]
        va = 'bottom' if values[i] > 0 else 'top'
        offset = 1.5 if values[i] > 0 else -1.5
        ax.text(i, y + offset * 0.5, f"{values[i]:+.2f}%", ha='center', va=va, fontsize=8, zorder=4)

    y_max = max(starts + values) * 1.3  # aumenta espaço topo
    y_min = min(starts + values)*1.2 if min(starts + values)<0 else 0
    ax.set_ylim(bottom=y_min, top=y_max)

    # Linha da predição: desenha só no topo da última barra (última label)
    last_bar_pos = len(values) - 1
    ax.hlines(y=prediction_pct, xmin=last_bar_pos - 0.4, xmax=last_bar_pos + 0.4,
              colors='black', linestyles='--', linewidth=1, alpha=0.7, zorder=1)

    # # Texto da predição fora do gráfico (acima do topo)
    # ax.text(last_bar_pos, y_max * 1.02, f'Predição média: {prediction_pct:.2f}%',
    #         ha='center', va='bottom', fontsize=9, color='black', zorder=5)

    ax.set_ylabel('Contribuição média SHAP (%)')
    ax.set_title(f'SHAP Waterfall Percentual - Campanha: {campanha_escolhida}')
    # ax.legend([f'Predição média: {prediction_pct:.2f}%'])
    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.show()

    # Salva imagem no buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    # Converte para imagem PIL
    image = Image.open(buf)
    return image