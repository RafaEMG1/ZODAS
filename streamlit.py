# Cargue de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import kagglehub
import os
import altair as alt
import plotly.express as px
import prince
import mca
import warnings

from sklearn import set_config
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score
)
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    mutual_info_classif,
    f_classif,
    f_regression,
    mutual_info_regression,
    RFE,
    RFECV
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Perceptron
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error
)
from scipy.stats import uniform, randint
from graphviz import Digraph

# Ignorar advertencias
warnings.filterwarnings("ignore")


st.set_page_config(page_title="Cirrosis Hepatica Streamlit App", layout="wide")
st.title("Clasificación de los estadios de la cirrosis hepática con métodos de Machine Learning")


st.header("Equipo")
st.write("[Diego Fernando Naranjo Polanía](mailto:dnaranjop@unbosque.edu.co)")
st.write("[Johan Steven Mahecha Cobos](mailto:jsmahecha@unbosque.edu.co)")
st.write("[Rafael Eduardo Montenegro González](mailto:rmontenegro@unbosque.edu.co)")

st.subheader("Origen de la base de datos")
st.write("[Kaggle – Liver Cirrhosis Stage Classification](https://www.kaggle.com/datasets/aadarshvelu/liver-cirrhosis-stage-classification)")
st.subheader("Colab:")
st.write("[Colab del proyecto](https://colab.research.google.com/drive/1-HsrnyL8eBXUKPOskzmHl3-udO2dWQc7?usp=sharing)")


# ----------------------------
# Sección de Metodología
# ----------------------------
st.title("🧪 Metodología del Proyecto")

st.markdown("""
Este proyecto sigue una **metodología de Machine Learning** para la clasificación de la cirrosis hepática.  
A continuación, se presentan los pasos de manera interactiva:
""")

# Paso 1
with st.expander("📌 Paso 1: Carga y Análisis Exploratorio de Datos"):
    st.write("""
    - Se utilizó un dataset con información clínica de pacientes.  
    - El archivo fue almacenado en GitHub y cargado en streamlit.  
    - Se revisó la calidad de los datos para identificar valores nulos
    - Se crean dos secciones con filtros para revisar las variables categóricas y numéricas.
    """)

# Paso 2
with st.expander("📌 Paso 2: Preprocesamiento"):
    st.write("""
    - Limpieza de datos: imputación de valores faltantes.  
    - Codificación de variables categóricas (One-Hot Encoding).  
    - Estandarización de las variables numéricas.  
    """)

# Paso 3
with st.expander("📌 Paso 3: Selección de características"):
    st.write("""
    - Se utilizaron técnicas de filtrado de variables como: 
        - Variables categóricas: $\chi^2$ e información mutua
        - Variables numéricas: ANOVA e información mutua
    - MCA y PCA
    - RFE (Recursive Feature Elimination) con validación cruzada (selección por envoltura)
    - Esto permite quedarnos solo con las variables más relevantes para el modelo.  
    """)

# Paso 4
with st.expander("📌 Paso 4: Entrenamiento del modelo"):
    st.write("""
    - Se probaron algoritmos como:
        - **Decission tree**
        - **Regresión Logística**
        - **Random forest**
        - **KNN (K-Nearest Neighbors)**
        - **SVM (Support Vector Machine)**
        - **HistGradientBoosting**
        - **ExtraTrees**
    """)

# Paso 5
with st.expander("📌 Paso 5: Búsqueda de mejores hiperparametros"):
    st.write("""
    - Randomized Search
    """)
    
# Paso 6
with st.expander("📌 Paso 6: Evaluación"):
    st.write("""
    - Se calcularon métricas como **Accuracy**, Precision, Recall y F1-Score.  
    - También se aplicó validación cruzada para obtener una estimación más robusta.  
    """)

st.subheader("🔎 Flujo Metodológico")

dot = Digraph()

dot.node("A", "Carga de Datos", shape="box")
dot.node("B", "Preprocesamiento", shape="box")
dot.node("C", "Selección de características", shape="box")
dot.node("D", "Entrenamiento del modelo\n", shape="box")
dot.node("E", "Búsqueda de mejores hiperparametros", shape="box")
dot.node("F", "Evaluación del modelo\n **Accuracy**", shape="box")

dot.edges(["AB", "BC", "CD", "DE","EF"])

st.graphviz_chart(dot)

st.caption("Estudio clínico de cirrosis hepática — ficha de variables")

texto = """
### **Variables:**

* **N_Days**: Número de días transcurridos entre el registro y la fecha más temprana entre fallecimiento, trasplante o análisis del estudio en 1986.  
* **Status**: estado del paciente C (censurado), CL (censurado por tratamiento hepático) o D (fallecimiento).  
* **Drug**: tipo de fármaco: D-penicilamina o placebo.  
* **Age**: edad en días.  
* **Sex**: M (hombre) o F (mujer).  
* **Ascites**: presencia de ascitis N (No) o Y (Sí).  
* **Hepatomegaly**: presencia de hepatomegalia N (No) o Y (Sí).  
* **Spiders**: presencia de aracnosis N (No) o Y (Sí).  
* **Edema**: presencia de edema N (sin edema ni tratamiento diurético), S (edema presente sin diuréticos o resuelto con diuréticos) o Y (edema a pesar del tratamiento diurético).  
* **Bilirubin**: bilirrubina sérica en mg/dl.  
* **Cholesterol**: colesterol sérico en mg/dl.  
* **Albumin**: albúmina en g/dl.  
* **Copper**: cobre en orina en µg/día.  
* **Alk_Phos**: fosfatasa alcalina en U/litro.  
* **SGOT**: SGOT en U/ml.  
* **Tryglicerides**: triglicéridos en mg/dl.  
* **Platelets**: plaquetas por metro cúbico [ml/1000].  
* **Prothrombin**: tiempo de protrombina en segundos.  
* **Stage**: estadio histológico de la enfermedad (1, 2 o 3).  

---

### **Dimensiones del dataset**
- **Tamaño:** 25 000 filas, 19 columnas  
- **Faltantes:** 0% en todas las columnas  

---
"""

st.markdown(texto)

# --- Cargar dataset ---
# URL del CSV en GitHub (raw)
url = "https://raw.githubusercontent.com/DiegoNaranjo84/cirrosis_hepatica/main/liver_cirrhosis.csv"

# Cargar el dataset
df = pd.read_csv(url)

# Filtrar solo columnas categóricas (tipo "object" o "category")
df['Stage'] = pd.to_numeric(df['Stage'], errors='coerce')
df['Stage'] = pd.Categorical(df['Stage'], ordered=True)
cat_cols = df.select_dtypes(include=['object', 'category'])

st.subheader("Primeras 10 filas del dataset")
st.dataframe(df.head(10), use_container_width=True)

# ------- Helpers -------
def format_uniques(series, max_items=20):
    """Convierte valores únicos a una cadena legible, acota a max_items."""
    uniques = pd.Series(series.dropna().unique())
    head = uniques.head(max_items).astype(str).tolist()
    txt = ", ".join(head)
    if uniques.size > max_items:
        txt += f" … (+{uniques.size - max_items} más)"
    return txt

# ------- Detectar tipos -------
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df.select_dtypes(include=["number"]).columns.tolist()

# ------- Resumen variables categóricas -------
cat_summary = pd.DataFrame({
    "Variable": cat_cols,
    "Tipo de dato": [df[c].dtype for c in cat_cols],
    "Nº de categorías únicas": [df[c].nunique(dropna=True) for c in cat_cols],
    "Nº de datos no nulos": [df[c].notna().sum() for c in cat_cols],
    "Categorías": [format_uniques(df[c], max_items=20) for c in cat_cols],
})

# ------- Resumen variables numéricas -------
num_summary = pd.DataFrame({
    "Variable": num_cols,
    "Tipo de dato": [df[c].dtype for c in num_cols],
    "Nº de datos no nulos": [df[c].notna().sum() for c in num_cols],
    "Mínimo": [df[c].min(skipna=True) for c in num_cols],
    "Máximo": [df[c].max(skipna=True) for c in num_cols],
    "Media":  [df[c].mean(skipna=True) for c in num_cols],
    "Desviación estándar": [df[c].std(skipna=True) for c in num_cols],
}).round(2)

# ------- Mostrar en dos columnas iguales con separación uniforme -------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Resumen variables categóricas")
    st.dataframe(cat_summary, use_container_width=True)

with col2:
    st.subheader("Resumen variables numéricas")
    st.dataframe(num_summary, use_container_width=True)

#####--------------------------------------------------------------------------------------#########

st.markdown("""### Análisis de variables categóricas""")
st.caption("Selecciona una variable para ver su distribución en tabla y gráfico de torta.")

variables_categoricas = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

if not variables_categoricas:
    st.warning("No se detectaron variables categóricas (object/category/bool) en df.")
    st.stop()

# =========================
# Controles (En la sección)
# =========================
st.markdown("*Controles*")
with st.container():
    c1, c2 = st.columns([1.6, 1.1])
    with c1:
        var = st.selectbox(
            "Variable categórica",
            options=variables_categoricas,
            index=0,
            key="cat_var_local"
        )
        top_n = st.slider(
            "Top N (agrupa el resto en 'Otros')",
            min_value=3, max_value=30, value=10, step=1,
            help="Agrupa las categorías menos frecuentes en 'Otros'",
            key="cat_topn_local"
        )
    with c2:
        orden_alfabetico = st.checkbox("Ordenar alfabéticamente (solo tabla)", value=False, key="cat_orden_local")

# =========================
# Preparar datos 
# =========================
serie = df[var].copy()

vc = serie.value_counts(dropna=True)

# Etiqueta amigable para NaN
labels = vc.index.to_list()
labels = ["(NaN)" if (isinstance(x, float) and np.isnan(x)) else str(x) for x in labels]
counts = vc.values

data = pd.DataFrame({"Categoría": labels, "Conteo": counts})
data["Porcentaje"] = (data["Conteo"] / data["Conteo"].sum() * 100).round(2)

# Agrupar en "Otros" si supera Top N
if len(data) > top_n:
    top = data.nlargest(top_n, "Conteo").copy()
    otros = data.drop(top.index)
    fila_otros = pd.DataFrame({
        "Categoría": ["Otros"],
        "Conteo": [int(otros["Conteo"].sum())],
        "Porcentaje": [round(float(otros["Porcentaje"].sum()), 2)]
    })
    data_plot = pd.concat([top, fila_otros], ignore_index=True)
else:
    data_plot = data.copy()

# Orden por Conteo (siempre)
data_plot = data_plot.sort_values("Conteo", ascending=False).reset_index(drop=True)

# Orden opcional alfabético en la tabla (no afecta el gráfico)
data_table = data_plot.copy()
if orden_alfabetico:
    data_table = data_table.sort_values("Categoría").reset_index(drop=True)

# =========================
# Mostrar tabla y gráfico
# =========================
tcol, gcol = st.columns([1.1, 1.3], gap="large")

with tcol:
    st.subheader(f"Distribución de {var}")
    st.dataframe(
        data_table.assign(Porcentaje=data_table["Porcentaje"].round(2)),
        use_container_width=True
    )

with gcol:
    st.subheader("Gráfico de torta")
    chart = (
        alt.Chart(data_plot)
        .mark_arc(outerRadius=110)
        .encode(
            theta=alt.Theta(field="Conteo", type="quantitative"),
            color=alt.Color("Categoría:N", legend=alt.Legend(title="Categoría")),
            tooltip=[
                alt.Tooltip("Categoría:N"),
                alt.Tooltip("Conteo:Q", format=","),
                alt.Tooltip("Porcentaje:Q", format=".2f")
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)

# =========================
# Extras informativos
# =========================
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.metric("Categorías mostradas", f"{len(data_plot)}")
with c2:
    st.metric("Total registros (variable seleccionada)", f"{int(serie.shape[0]):,}".replace(",", "."))

st.caption("Consejo: usa *Top N* para simplificar la lectura y agrupar categorías poco frecuentes en 'Otros'.")

#####--------------------------------------------------------------------------------------#########




# =========================
# Análisis de variables numéricas
# =========================
st.markdown("""### Análisis de variables numéricas""")
st.caption("Selecciona una variable para ver su distribución en tabla, boxplot e histograma.")

# Detectar variables numéricas
variables_numericas = df.select_dtypes(include=["number"]).columns.tolist()

if not variables_numericas:
    st.warning("No se detectaron variables numéricas en df.")
    st.stop()

# =========================
# Controles dentro de la sección
# =========================
st.markdown("*Controles*")
with st.container():
    c1, c2 = st.columns([1.6, 1.4])
    with c1:
        var_num = st.selectbox(
            "Variable numérica",
            options=variables_numericas,
            index=0,
            key="num_var_local"
        )
    with c2:
        bins = st.slider(
            "Número de bins (histograma)",
            min_value=5, max_value=100, value=30, step=5,
            key="num_bins_local"
        )

# =========================
# Preparar serie
# =========================
serie_num = df[var_num].dropna()

# =========================
# Métricas descriptivas
# =========================
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Nº datos no nulos", f"{serie_num.shape[0]:,}".replace(",", "."))
with c2:
    st.metric("Mínimo", f"{serie_num.min():.2f}")
with c3:
    st.metric("Máximo", f"{serie_num.max():.2f}")
with c4:
    st.metric("Media", f"{serie_num.mean():.2f}")
with c5:
    st.metric("Desv. Estándar", f"{serie_num.std():.2f}")

# =========================
# Gráficos
# =========================
g1, g2 = st.columns(2, gap="large")

# --- Boxplot vertical y ancho ---
with g1:
    st.subheader(f"Boxplot de {var_num} ")
    box_data = pd.DataFrame({var_num: serie_num})
    box_data["_grupo_"] = "Distribución"  # ancla un grupo único en X

    box_chart = (
        alt.Chart(box_data)
        .mark_boxplot(size=140, extent=1.5)  # size = ancho de la caja; extent=1.5 => whiskers tipo Tukey
        .encode(
            x=alt.X("_grupo_:N", axis=None, title=""),
            y=alt.Y(f"{var_num}:Q", title=var_num)
        )
        .properties(height=350)
    )
    st.altair_chart(box_chart, use_container_width=True)

# --- Histograma ---
with g2:
    st.subheader(f"Histograma de {var_num}")
    hist_data = pd.DataFrame({var_num: serie_num})
    hist_chart = (
        alt.Chart(hist_data)
        .mark_bar()
        .encode(
            alt.X(var_num, bin=alt.Bin(maxbins=bins)),
            y='count()',
            tooltip=[
                alt.Tooltip(var_num, bin=alt.Bin(maxbins=bins)),
                alt.Tooltip('count()', title="Frecuencia")
            ]
        )
        .properties(height=350)
    )
    st.altair_chart(hist_chart, use_container_width=True)



st.subheader("Distribuciones Simétricas")
st.markdown("""
Las variables **Age**, **Albumin**, **Platelets** y **Prothrombin** presentan distribuciones cercanas a la forma de campana, 
lo que indica una dispersión relativamente equilibrada alrededor de la media.  
Aunque existe una ligera asimetría, la mayoría de los valores se concentran en torno a la mediana y la media, 
sin presencia marcada de colas largas.
""")

st.subheader("Distribuciones con Sesgo a la Derecha (Asimetría Positiva)")
st.markdown("""
Las variables **Bilirubin**, **Cholesterol**, **Copper**, **Alk_Phos**, **SGOT**, **Triglycerides** y **N_Days** muestran 
**asimetría positiva**, es decir, presentan una concentración de valores en la parte baja del rango y colas largas hacia la derecha.  
Esto sugiere alta dispersión y la posible presencia de valores extremos (outliers), particularmente en variables bioquímicas:

- **Bilirubin**: valores entre 0,3 y 28 mg/dL, media de 3,40 mg/dL  
- **Cholesterol**: promedio de 372 mg/dL con máxima dispersión (DE ≈ 193,7), hasta 1775 mg/dL  
- **Copper** y **Alk_Phos**: alta variabilidad, con Alk_Phos alcanzando 13862,4 U/L  
- **SGOT**: valores entre 26,35 y 457,25 U/L, media de 123,17 U/L  
- **Triglycerides**: promedio 123,82 mg/dL, con extremos hasta 598 mg/dL  
""")

st.subheader("Posibles Outliers Extremos")
st.markdown("""
Se identifican posibles valores atípicos en:

- **Bilirubin**: > 20 mg/dL  
- **Cholesterol**: > 1500 mg/dL  
- **Alk_Phos**: > 10000 U/L  
- **Copper**: > 500 µg/dL  
- **SGOT**: > 400 U/L  

Estos casos deben evaluarse para decidir si requieren tratamiento previo al modelado estadístico.
""")

st.subheader("Resumen de Variables Estables")
st.markdown("""
- **Albumin**: valores entre 1,96 y 4,64 g/dL, mediana 3,51 g/dL, baja dispersión  
- **Platelets**: promedio 256 × 10³/µL, rango 62 a 721 × 10³/µL  
- **Prothrombin**: media de 10,73 s, variación reducida (DE ≈ 0,90 s)  

En conjunto, estas variables muestran un comportamiento más estable y menos afectado por valores extremos.
""")



# =========================
# Matriz de Correlación (media pantalla)
# =========================
st.markdown("### Matriz de Correlación")
correlacion = df.corr(numeric_only=True)

import seaborn as sns
import matplotlib.pyplot as plt

# Tamaño fijo para ocupar ~media pantalla
fig_w, fig_h = 8, 4  # ancho=8, alto=4 pulgadas (antes era más alto)

with sns.plotting_context("notebook", font_scale=0.6):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    hm = sns.heatmap(
        correlacion,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
        annot_kws={"size": 5},       # texto más pequeño
        cbar_kws={"shrink": 0.4},    # barra de color más compacta
        linewidths=0.3,
        linecolor="white"
    )

    # Título y etiquetas pequeñas
    ax.set_title("Matriz de Correlación", fontsize=9, pad=4)
    ax.tick_params(axis="x", labelsize=5, rotation=45)
    ax.tick_params(axis="y", labelsize=5)

    # Barra de color con etiquetas pequeñas
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5)

    fig.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)  # Se ajusta al ancho del contenedor


# ============================
# Análisis de correlaciones en Streamlit
# ============================

st.subheader("Correlaciones Positivas Más Destacadas")
st.markdown("""
- **Bilirubin y Copper:** 0.43 → Correlación positiva moderada; niveles más altos de bilirrubina tienden a coincidir con más cobre.  
- **Bilirubin y SGOT:** 0.37  
- **Bilirubin y Triglycerides:** 0.38 → Relaciones positivas moderadas.  
- **Albumin y N_Days:** 0.37 → Mayor albúmina asociada con más días de seguimiento.  
- **SGOT y Cholesterol:** 0.32  
- **SGOT y Copper:** 0.28 → Ligera asociación de la enzima SGOT con colesterol y cobre.  
- **Bilirubin y Cholesterol:** 0.34 → Relación positiva moderada.
""")

st.subheader("Correlaciones Negativas Relevantes")
st.markdown("""
- **Bilirubin y N_Days:** -0.39 → Mayor bilirrubina, menor tiempo de supervivencia/seguimiento.  
- **Copper y N_Days:** -0.28 → Tendencia a menor tiempo de seguimiento con cobre alto.  
- **Albumin y Bilirrubin:** -0.28 → Más bilirrubina asociada con menos albúmina, consistente con disfunción hepática.
""")

st.subheader("Análisis General")
st.markdown("""
La **bilirrubina** es la variable con mayores correlaciones:  
- Positivas moderadas con cobre, SGOT y colesterol  
- Negativas con tiempo de seguimiento y albúmina  

No se observan correlaciones fuertes (> 0.7), lo que sugiere que las variables pueden aportar **información complementaria** en futuros modelos.
""")


#________________________________________________________________________________________________________________________________________________________________




# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""# 1. Selección de carácteristicas""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 1.1. Selección de carácteristicas categóricas""")

# =========================
# Preparación de datos
# =========================
# Filtramos solo categóricas y dejamos Stage como y (objetivo)
df_cat = df.select_dtypes(include=["object", "category", "bool"]).copy()
if "Stage" not in df.columns:
    st.error("❌ No se encontró la columna objetivo 'Stage'.")
else:
    # Asegurar que Stage esté como y y no en X
    y_cat = df["Stage"]
    X_cat = df_cat.drop(columns=[c for c in ["Stage"] if c in df_cat.columns], errors="ignore")

    if X_cat.shape[1] == 0:
        st.info("No hay variables categóricas (distintas a 'Stage') para evaluar.")
    else:
        # Split estratificado
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_cat, y_cat, test_size=0.33, random_state=1, stratify=y_cat
        )

        # One-Hot Encoding (una sola vez y la reusamos)
        ohe_11 = OneHotEncoder(handle_unknown="ignore", sparse=True)
        X_train_ohe = ohe_11.fit_transform(X_train_c)
        X_test_ohe  = ohe_11.transform(X_test_c)
        feature_names_11 = ohe_11.get_feature_names_out(X_cat.columns)

        # LabelEncoder para y
        le_11 = LabelEncoder()
        y_train_enc_11 = le_11.fit_transform(y_train_c)
        y_test_enc_11  = le_11.transform(y_test_c)

        # -------------------------
        # Controles de la sección
        # -------------------------
        cA, cB, cC = st.columns([1.2, 1, 1])
        thr_pct = cA.slider(
            "Umbral de cobertura (porcentaje acumulado)",
            min_value=50, max_value=99, value=90, step=1, key="cat11_thr"
        )
        top_n_plot = cB.slider(
            "Top-N para el gráfico",
            min_value=10, max_value=200, value=40, step=5, key="cat11_topn"
        )
        mostrar_tabla_completa = cC.checkbox(
            "Mostrar tabla completa de dummies", value=False, key="cat11_tabla_full"
        )

        st.caption("Selecciona método de puntuación para ordenar dummies:")
        tab_chi2, tab_mi = st.tabs(["χ² (Chi-cuadrado)", "Información Mutua"])

        # =========================
        # Helper para ejecutar y mostrar resultados
        # =========================
        def run_selector(score_func, titulo, key_prefix):
            # SelectKBest con k='all' solo para obtener *todas* las puntuaciones
            selector = SelectKBest(score_func=score_func, k="all")
            selector.fit(X_train_ohe, y_train_enc_11)
            scores = selector.scores_

            # Proteger contra NaN o None
            scores = np.nan_to_num(scores, nan=0.0)

            # Orden descendente
            idx = np.argsort(scores)[::-1]
            sorted_scores = scores[idx]
            sorted_feats  = feature_names_11[idx]

            # Porcentaje acumulado
            total = np.sum(sorted_scores) if np.sum(sorted_scores) > 0 else 1.0
            cum = np.cumsum(sorted_scores) / total
            cutoff_idx = int(np.searchsorted(cum, thr_pct / 100.0) + 1)
            selected = sorted_feats[:cutoff_idx]

            # Métricas
            c1, c2, c3 = st.columns(3)
            c1.metric("Dummies totales", f"{len(feature_names_11)}")
            c2.metric("Seleccionadas", f"{cutoff_idx}")
            c3.metric("Umbral", f"{thr_pct}%")

            # Tabla
            df_scores = pd.DataFrame({
                "Dummy (OHE)": sorted_feats,
                "Score": np.round(sorted_scores, 6),
                "Acumulado": np.round(cum, 4)
            })
            if not mostrar_tabla_completa:
                st.dataframe(df_scores.head(top_n_plot), use_container_width=True)
            else:
                st.dataframe(df_scores, use_container_width=True)

            # Gráfico barras Top-N
            fig, ax = plt.subplots(figsize=(10, 4))
            n_plot = min(top_n_plot, len(sorted_feats))
            ax.bar(range(n_plot), sorted_scores[:n_plot])
            ax.set_xticks(range(n_plot))
            ax.set_xticklabels(sorted_feats[:n_plot], rotation=90)
            ax.set_ylabel("Puntuación")
            ax.set_title(f"{titulo} — Top-{n_plot}")
            ax.axvline(cutoff_idx - 1, color="red", linestyle="--", label=f"Umbral {thr_pct}%")
            ax.legend(loc="upper right")
            fig.tight_layout()
            st.pyplot(fig)

            # Resumen y lista seleccionadas
            with st.expander("📄 Variables seleccionadas (hasta el umbral)"):
                st.write(selected.tolist())

        # =========================
        # Pestaña χ²
        # =========================
        with tab_chi2:
            st.markdown("**Método:** χ² (para asociación categórica vs. clases del objetivo)")
            run_selector(chi2, "SelectKBest χ²", "chi2_11")

        # =========================
        # Pestaña Información Mutua
        # =========================
        with tab_mi:
            st.markdown("**Método:** Información Mutua (dependencia no lineal)")
            run_selector(mutual_info_classif, "SelectKBest Información Mutua", "mi_11")


st.subheader("Conclusión Parcial: Chi² e Información Mutua")
st.markdown("""
Tras aplicar las técnicas de **Chi²** e **Información Mutua**, ambos métodos convergieron en el mismo conjunto de **9 variables categóricas**,  
con evidencia de **aporte predictivo** y coherencia.  

Dado que son variables nominales, la **codificación One-Hot-Encoder (OHE)** es la alternativa recomendada, pasando de **7 variables iniciales a 16 codificadas**.

Al coincidir los dos criterios de selección, los modelos se ejecutaron una sola vez sobre este conjunto.  
En los experimentos con solo variables categóricas:  
- **Random Forest** y **Árboles de Decisión** lograron la mejor *accuracy* ≈ **51,62 %**  

Este resultado se interpreta como una **cota inferior del desempeño**, ya que se espera una mejora al **integrar las variables numéricas**
""")


# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 1.2. Selección de carácteristicas numéricas""")


# =========================
# 1.2. Selección de características numéricas
# =========================

# --- Detectar y preparar numéricas ---
df_num_full = df.select_dtypes(include=["number"]).copy()

# Asegura que Stage no esté en X
num_cols_12 = [c for c in df_num_full.columns if c != "Stage"]

if len(num_cols_12) == 0:
    st.info("No hay variables numéricas (distintas a 'Stage') para evaluar.")
else:
    X_num = df[num_cols_12].copy()
    y_num = df["Stage"].copy()

    # Split estratificado
    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
        X_num, y_num, test_size=0.33, random_state=1, stratify=y_num
    )

    # y codificada para los score_func
    le_12 = LabelEncoder()
    y_train_enc_12 = le_12.fit_transform(y_train_n)
    y_test_enc_12  = le_12.transform(y_test_n)

    # -------------------------
    # Controles
    # -------------------------
    cA, cB, cC = st.columns([1.2, 1, 1])
    thr_pct_n = cA.slider(
        "Umbral de cobertura (porcentaje acumulado)",
        min_value=50, max_value=99, value=90, step=1, key="num12_thr"
    )
    top_n_plot_n = cB.slider(
        "Top-N para el gráfico",
        min_value=5, max_value=50, value=20, step=1, key="num12_topn"
    )
    show_full_tbl_n = cC.checkbox(
        "Mostrar tabla completa", value=False, key="num12_tbl_full"
    )

    tab_anova, tab_mi = st.tabs(["ANOVA (f_classif)", "Información Mutua"])

    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

    def run_numeric_selector(score_func, titulo, key_prefix):
        # k='all' para obtener todas las puntuaciones
        selector = SelectKBest(score_func=score_func, k="all")
        selector.fit(X_train_n, y_train_enc_12)

        scores = selector.scores_
        # Proteger NaN (puede pasar si una columna es constante)
        scores = np.nan_to_num(scores, nan=0.0)

        # Orden descendente
        idx = np.argsort(scores)[::-1]
        sorted_scores = scores[idx]
        sorted_feats  = np.array(num_cols_12)[idx]

        # Porcentaje acumulado
        total = np.sum(sorted_scores) if np.sum(sorted_scores) > 0 else 1.0
        cum = np.cumsum(sorted_scores) / total
        cutoff_idx = int(np.searchsorted(cum, thr_pct_n / 100.0) + 1)
        selected = sorted_feats[:cutoff_idx]

        # Métricas rápidas
        c1, c2, c3 = st.columns(3)
        c1.metric("Variables numéricas", f"{len(num_cols_12)}")
        c2.metric("Seleccionadas", f"{cutoff_idx}")
        c3.metric("Umbral", f"{thr_pct_n}%")

        # Tabla
        df_scores = pd.DataFrame({
            "Variable": sorted_feats,
            "Score": np.round(sorted_scores, 6),
            "Acumulado": np.round(cum, 4)
        })
        if show_full_tbl_n:
            st.dataframe(df_scores, use_container_width=True)
        else:
            st.dataframe(df_scores.head(top_n_plot_n), use_container_width=True)

        # Gráfico Top-N
        fig, ax = plt.subplots(figsize=(10, 4))
        n_plot = min(top_n_plot_n, len(sorted_feats))
        ax.bar(range(n_plot), sorted_scores[:n_plot])
        ax.set_xticks(range(n_plot))
        ax.set_xticklabels(sorted_feats[:n_plot], rotation=90)
        ax.set_ylabel("Puntuación")
        ax.set_title(f"{titulo} — Top-{n_plot}")
        ax.axvline(cutoff_idx - 1, color="red", linestyle="--", label=f"Umbral {thr_pct_n}%")
        ax.legend(loc="upper right")
        fig.tight_layout()
        st.pyplot(fig)

        # Lista seleccionadas
        with st.expander("📄 Variables seleccionadas (hasta el umbral)"):
            st.write(selected.tolist())

    with tab_anova:
        st.markdown("**Método:** ANOVA (f_classif) — relación lineal con clases.")
        run_numeric_selector(f_classif, "SelectKBest ANOVA (f_classif)", "anova")

    with tab_mi:
        st.markdown("**Método:** Información Mutua — dependencias no lineales.")
        run_numeric_selector(mutual_info_classif, "SelectKBest Información Mutua", "mi")

    # -----------------------------------------------------------------
    # (Opcional) Buscar k óptimo con CV y Modelo (LogReg) — liviano
    # -----------------------------------------------------------------
    with st.expander("🔎 (Opcional) Buscar k óptimo con validación cruzada"):
        st.caption("Se prueba k=1..N con Regresión Logística multinomial. Incluye StandardScaler.")
        run_cv = st.checkbox("Ejecutar búsqueda de k", value=False, key="num12_cv_run")
        metodo_cv = st.selectbox(
            "Score function para SelectKBest",
            options=["ANOVA (f_classif)", "Información Mutua"],
            index=0, key="num12_cv_sf"
        )
        if run_cv:
            score_func_cv = f_classif if metodo_cv == "ANOVA (f_classif)" else mutual_info_classif
            from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
            pipe = Pipeline(steps=[
                ("scaler", StandardScaler()),
                ("kbest", SelectKBest(score_func=score_func_cv)),
                ("clf", LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=2000))
            ])
            param_grid = {"kbest__k": list(range(1, len(num_cols_12) + 1))}
            search = GridSearchCV(pipe, param_grid, scoring="accuracy", n_jobs=-1, cv=cv)
            search.fit(X_train_n, y_train_n)
            st.write(f"**Mejor k:** {search.best_params_['kbest__k']}")
            st.write(f"**Mejor Accuracy CV:** {search.best_score_:.4f}")


st.subheader("Análisis")

st.markdown("""
- Los dos selectores de características f_classif y mutual info classif presentan diferentes scores y seleccionan distintas variables, aunque coinciden en las características N_Dias, Colesterol, y triglicéridos. 
- Con mutual_info_classif la mayoría de las variables muestran un score más representativo, destacando las características N_Dias, Edad y Plaquetas., 
- Con f_classif las características más representativas son Prothrombin, N_Dias y Albumin.
Esta diferencia no permitió definir un conjunto único de características para trabajar de forma confiable. Por ello, se decidió modelar y observar el comportamiento con las diferentes variables seleccionadas.
Los diferentes modelos no mejoraron el accuracy respecto al modelo principal con todas las variables.

- En la regresión logística softmax, con 7 variables se obtuvo un accuracy similar al del modelo completo, sin embargo, el modelo no presenta un buen accuracy.
- Los algoritmos utilizados no permitieron reducir de forma efectiva la cantidad de variables sin afectar el desempeño.
- Con solo variables numéricas, el modelo SVC alcanzó el mejor accuracy (≈ 0.7557) sin reducir variables.
Se espera que este resultado mejore al integrar las variables categóricas.
""")





# __________________________________________________________________________________________________
st.markdown("## 1.3. Unión de variables categóricas y numéricas")

# ====== CONFIG LOCAL DE LA SECCIÓN (no global) ======
s13_ohe_keep = [
    "Hepatomegaly_N", "Hepatomegaly_Y",
    "Status_D", "Status_C",           # si existe 'Status_CL' quedará fuera a propósito
    "Edema_N", "Edema_S", "Edema_Y",
    "Spiders_Y", "Spiders_N",
]

# --- Selección de columnas desde df sin modificarlo ---
s13_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
s13_cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
for _L in (s13_num_cols, s13_cat_cols):
    if "Stage" in _L:
        _L.remove("Stage")

# --- Construcción de X/y LOCALES (copias) ---
s13_X = df[s13_num_cols + s13_cat_cols].copy()
s13_y = df["Stage"].copy()

# === Compatibilidad OHE (sparse_output vs sparse) ===
try:
    s13_OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    s13_OHE = OneHotEncoder(handle_unknown="ignore", sparse=False)

# --- Helpers que NO mutan y que son locales a 1.3 ---
def s13__endswith_any(colname: str, suffixes: list[str]) -> bool:
    return any(str(colname).endswith(suf) for suf in suffixes)

def s13_select_keep_cols(X_df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona columnas OHE objetivo tolerando prefijos como 'cat__' o 'ohe__'.
    No muta X_df; crea columnas faltantes en una copia.
    """
    Xc = X_df.copy()
    out_cols = []
    for target in s13_ohe_keep:
        matches = [c for c in Xc.columns if s13__endswith_any(c, [target])]
        if matches:
            out_cols.extend(matches)
        else:
            # crear columna faltante en copia
            Xc[target] = 0
            out_cols.append(target)
    return Xc[out_cols]

def s13_keep_feature_names(_, input_features):
    # Devolver exactamente los nombres esperados; estable y local
    return np.array(s13_ohe_keep, dtype=object)

# === Pipelines LOCALES ===
s13_num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

s13_cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", s13_OHE),
    ("select", FunctionTransformer(s13_select_keep_cols, feature_names_out=s13_keep_feature_names))
])

s13_preprocess = ColumnTransformer(
    transformers=[
        ("num", s13_num_pipe, s13_num_cols),
        ("cat", s13_cat_pipe, s13_cat_cols)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# >>> IMPORTANTÍSIMO: salida a pandas SOLO en este objeto (no set_config global)
try:
    s13_preprocess.set_output(transform="pandas")
except Exception:
    # En versiones antiguas de sklearn no existe set_output; seguirá devolviendo np.ndarray
    pass

# === Split LOCAL ===
s13_X_train, s13_X_test, s13_y_train, s13_y_test = train_test_split(
    s13_X, s13_y, test_size=0.33, random_state=42, stratify=s13_y
)

# === Fit/Transform LOCAL ===
s13_X_train_t = s13_preprocess.fit_transform(s13_X_train)
s13_X_test_t  = s13_preprocess.transform(s13_X_test)

# === OUTPUT SOLO DE LA SECCIÓN 1.3 ===
st.markdown("**Vista rápida del dataset transformado (Sección 1.3 aislada)**")
c1, c2 = st.columns(2)
with c1:
    st.write("**Train shape**:", s13_X_train_t.shape)
    st.dataframe(
        s13_X_train_t.head(8) if isinstance(s13_X_train_t, pd.DataFrame)
        else pd.DataFrame(s13_X_train_t).head(8),
        use_container_width=True
    )
with c2:
    st.write("**Test shape**:", s13_X_test_t.shape)
    st.dataframe(
        s13_X_test_t.head(8) if isinstance(s13_X_test_t, pd.DataFrame)
        else pd.DataFrame(s13_X_test_t).head(8),
        use_container_width=True
    )

# __________________________________________________________________________________________________
st.markdown("## 1.4. Modelado ")

# ===== Helpers locales (no globales) =====
def s14_make_safe_cv(y_like, max_splits=5, seed=42):
    ys = pd.Series(y_like).dropna()
    min_class = ys.value_counts().min() if not ys.empty else 2
    n_splits = max(2, min(max_splits, int(min_class)))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

def s14_build_model(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(multi_class="multinomial", solver="lbfgs",
                                  max_iter=2000, class_weight="balanced", random_state=42)
    if name == "KNN":
        return KNeighborsClassifier()
    if name == "SVC":
        return SVC()  # ya venimos escalados desde 1.3 en numéricas
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    if name == "Random Forest":
        return RandomForestClassifier(random_state=42)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(random_state=42)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(random_state=42)
    raise ValueError("Modelo no soportado")
    
# ===== Reutilizar transformaciones de 1.3 si existen; si no, fallback mínimo =====
try:
    X_train_14 = s13_X_train_t
    X_test_14  = s13_X_test_t
    y_train_14 = s13_y_train
    y_test_14  = s13_y_test
except NameError:
    # Fallback: construir un preprocesado rápido similar a 1.3
    st.warning("No se encontraron objetos de la Sección 1.3; usando preprocesado mínimo local para 1.4.")
    # columnas
    s14_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    s14_cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    for _L in (s14_num_cols, s14_cat_cols):
        if "Stage" in _L:
            _L.remove("Stage")
    s14_X = df[s14_num_cols + s14_cat_cols].copy()
    s14_y = df["Stage"].copy()

    # OHE compatible
    try:
        s14_OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        s14_OHE = OneHotEncoder(handle_unknown="ignore", sparse=False)

    # pipelines
    s14_num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", StandardScaler())])
    s14_cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ("ohe", s14_OHE)])
    s14_pre = ColumnTransformer(
        [("num", s14_num_pipe, s14_num_cols),
         ("cat", s14_cat_pipe, s14_cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False
    )
    try:
        s14_pre.set_output(transform="pandas")
    except Exception:
        pass

    X_train_raw, X_test_raw, y_train_14, y_test_14 = train_test_split(
        s14_X, s14_y, test_size=0.33, random_state=42, stratify=s14_y
    )
    X_train_14 = s14_pre.fit_transform(X_train_raw)
    X_test_14  = s14_pre.transform(X_test_raw)

# ===== UI de selección de modelo =====
model_name_14 = st.selectbox(
    "Elige el modelo a evaluar",
    options=[
        "Logistic Regression",
        "KNN",
        "SVC",
        "Decision Tree",
        "Random Forest",
        "ExtraTrees",
        "HistGradientBoosting",
    ],
    index=0,
    key="s14_model_sel"
)

modelo_14 = s14_build_model(model_name_14)

# ===== CV seguro y evaluación =====
cv_14 = s14_make_safe_cv(y_train_14, max_splits=5)
scores_14 = cross_val_score(modelo_14, X_train_14, y_train_14, cv=cv_14, scoring="accuracy", n_jobs=-1)

st.subheader("Resultados de validación cruzada (1.4)")
st.write(f"**Modelo:** {model_name_14}")
st.write(f"**Accuracy (media CV):** {scores_14.mean():.4f}  |  **Std:** {scores_14.std():.4f}")

# ===== Entrenamiento final y métricas en Test =====
modelo_14.fit(X_train_14, y_train_14)
y_pred_14 = modelo_14.predict(X_test_14)

acc_test_14 = accuracy_score(y_test_14, y_pred_14)
st.markdown(f"**Accuracy Test:** {acc_test_14:.4f}")
st.text("📋 Classification Report (Test):")
st.text(classification_report(y_test_14, y_pred_14))

st.text("🧩 Matriz de Confusión (Test):")
st.write(pd.DataFrame(confusion_matrix(y_test_14, y_pred_14),
                      index=sorted(pd.unique(y_test_14)),
                      columns=sorted(pd.unique(y_test_14))))



# __________________________________________________________________________________________________
st.markdown("## 1.5. Búsqueda de hiperparámetros (Randomized Search)")

# Reutilizar datos procesados de la sección 1.3
try:
    X_train_15 = s13_X_train_t
    X_test_15  = s13_X_test_t
    y_train_15 = s13_y_train
    y_test_15  = s13_y_test
except NameError:
    st.error("❌ No se encontraron datos procesados de la Sección 1.3. Ejecuta esa sección primero.")
    st.stop()

# Selección del modelo para ajustar
model_name_15 = st.selectbox(
    "Elige el modelo para Randomized Search",
    options=[
        "Logistic Regression",
        "KNN",
        "SVC",
        "Decision Tree",
        "Random Forest",
        "ExtraTrees",
        "HistGradientBoosting"
    ],
    index=0,
    key="s15_model_sel"
)

# Espacios de búsqueda para cada modelo
def get_search_space(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=5000, multi_class="multinomial", solver="lbfgs"), {
            "C": uniform(0.01, 10)
        }
    if name == "KNN":
        return KNeighborsClassifier(), {
            "n_neighbors": randint(3, 30),
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"]
        }
    if name == "SVC":
        return SVC(), {
            "C": uniform(0.01, 10),
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"]
        }
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42), {
            "max_depth": randint(3, 20),
            "min_samples_split": randint(2, 10),
            "criterion": ["gini", "entropy"]
        }
    if name == "Random Forest":
        return RandomForestClassifier(random_state=42), {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 20),
            "min_samples_split": randint(2, 10),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2"]
        }
    if name == "ExtraTrees":
        return ExtraTreesClassifier(random_state=42), {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 20),
            "min_samples_split": randint(2, 10),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2"]
        }
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(random_state=42), {
            "max_iter": randint(50, 300),
            "max_depth": randint(3, 20),
            "learning_rate": uniform(0.01, 0.3),
            "min_samples_leaf": randint(1, 20)
        }

# Obtener modelo y espacio
model_15, param_dist = get_search_space(model_name_15)

# Validación cruzada
cv_15 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Randomized Search
rs = RandomizedSearchCV(
    estimator=model_15,
    param_distributions=param_dist,
    n_iter=20,
    cv=cv_15,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42
)

try:
    with st.spinner("⏳ Buscando mejores hiperparámetros..."):
        rs.fit(X_train_15, y_train_15)

    # Resultados si todo sale bien
    st.subheader("Mejores hiperparámetros encontrados")
    st.write(rs.best_params_)
    st.write(f"**Mejor Accuracy CV:** {rs.best_score_:.4f}")

    # Evaluación en Test
    best_model = rs.best_estimator_
    y_pred_15 = best_model.predict(X_test_15)

    st.markdown("### Evaluación en Test")
    st.write(f"**Accuracy Test:** {accuracy_score(y_test_15, y_pred_15):.4f}")
    st.text("📋 Classification Report:")
    st.text(classification_report(y_test_15, y_pred_15))

except Exception as e:
    st.error("⏰ Tiempo de procesamiento excedido o error en la ejecución. Por favor intente con otro modelo.")
    with st.expander("Detalles técnicos del error"):
        st.code(str(e))

def sec_tabla_completa():
    st.markdown("## 📊 Comparativa de Modelos con Métricas por Clase")

    # ======== Datos de todos los modelos ========
    data = [
        ("Logistic Regression", 0.5866, 0.5770,
         [(0.57, 0.60, 0.58), (0.51, 0.46, 0.48), (0.65, 0.67, 0.66)],
         (0.58, 0.58, 0.58)),
        ("KNN", 0.9315, 0.9265,
         [(0.92, 0.92, 0.92), (0.91, 0.92, 0.91), (0.95, 0.94, 0.95)],
         (0.93, 0.93, 0.93)),
        ("Decision Tree", 0.9071, 0.9070,
         [(0.90, 0.90, 0.90), (0.90, 0.90, 0.90), (0.93, 0.92, 0.92)],
         (0.91, 0.91, 0.91)),
        ("Random Forest", 0.9475, 0.9451,
         [(0.95, 0.94, 0.94), (0.93, 0.94, 0.93), (0.96, 0.96, 0.96)],
         (0.95, 0.95, 0.95)),
        ("ExtraTrees (CV)", 0.9409, 0.9356,
         [(0.93, 0.93, 0.93), (0.92, 0.92, 0.92), (0.96, 0.95, 0.95)],
         (0.94, 0.94, 0.94)),
        ("ExtraTrees (RandSearch)", 0.9310, 0.9273,
         [(0.93, 0.92, 0.92), (0.90, 0.92, 0.91), (0.95, 0.94, 0.95)],
         (0.93, 0.93, 0.93)),
        ("HistGradientBoosting", 0.9611, 0.9582,
         [(0.96, 0.95, 0.95), (0.95, 0.95, 0.95), (0.97, 0.97, 0.97)],
         (0.96, 0.96, 0.96)),
    ]

    # ======== Expandir a filas ========
    rows = []
    for modelo, acc_cv, acc_test, clases, macro in data:
        for i, (p, r, f) in enumerate(clases, start=1):
            rows.append({
                "Modelo": modelo,
                "Clase": i,
                "Precision": p,
                "Recall": r,
                "F1-score": f,
                "Macro Avg Precision": macro[0],
                "Macro Avg Recall": macro[1],
                "Macro Avg F1": macro[2],
                "Accuracy CV": acc_cv if i == 1 else np.nan,
                "Accuracy Test": acc_test if i == 1 else np.nan
            })

    df = pd.DataFrame(rows)

    # ======== Filtros y visualización ========
    modelos = ["Todos"] + [m for m, *_ in data]
    sel_modelo = st.selectbox("Filtrar por modelo:", options=modelos, index=0)

    if sel_modelo != "Todos":
        df_view = df[df["Modelo"] == sel_modelo].reset_index(drop=True)
    else:
        df_view = df

    st.dataframe(df_view, use_container_width=True)

sec_tabla_completa()



st.subheader("Análisis")
st.markdown("""
Con base en los resultados, se observa que los modelos simples como **Logistic Regression** quedan muy rezagados con un **accuracy** inferior al 60 %, mientras que **KNN** y **Decision Tree** logran entre el 90 % y 93 %. Los modelos de ensamble como **Random Forest** y **ExtraTrees** superan el 94 %, mostrando mejoras significativas frente a los modelos básicos. Sin embargo, el mejor desempeño lo alcanza **HistGradientBoosting**, con un **accuracy** cercano al 96 % en validación cruzada y prueba, además de métricas por clase (precisión, recall y F1-score) muy equilibradas, lo que evidencia **muy buena capacidad de generalización** y lo posiciona como la mejor alternativa para la clasificación multiclase en este problema.
""")




















# === INICIO SECCIÓN 2 (filtros en cada subsección) ===


st.markdown("# 2. MCA Y PCA")

# -----------------------------------------------------
# 2.1. MCA  (filtros dentro de la subsección)
# -----------------------------------------------------
st.markdown("## 2.1. MCA")

df_cat_all = df.select_dtypes(include=["object", "category", "bool"]).copy()
cat_cols_all = [c for c in df_cat_all.columns if c != "Stage"]

# Controles MCA (en línea, no sidebar)
mca_ctrl = st.container()
with mca_ctrl:
    c_m1, c_m2, c_m3 = st.columns([2, 1, 1])
    cat_sel = c_m1.multiselect(
        "Variables categóricas para MCA",
        options=cat_cols_all,
        default=cat_cols_all,
        key="mca_vars_sel_section"
    )
    var_target_mca = c_m2.slider(
        "Varianza objetivo (%)",
        min_value=80, max_value=99, value=80, step=1, key="mca_var_pct"
    ) / 100.0
    top_k_mca = c_m3.slider(
        "Top-K variables",
        min_value=5, max_value=50, value=15, step=1, key="mca_topk"
    )

if cat_sel:
    df_cat = df_cat_all[cat_sel]
    y = df["Stage"]
    X_train, X_test, y_train, y_test = train_test_split(
        df_cat, y, stratify=y, test_size=0.33, random_state=1
    )

    X_train_encoded = pd.get_dummies(X_train, drop_first=False)
    if X_train_encoded.shape[1] == 0:
        st.info("Selecciona al menos una variable categórica para ejecutar MCA.")
    else:
        mca_cirrosis = mca.MCA(X_train_encoded, benzecri=True)

        # Varianza acumulada
        sv = mca_cirrosis.s
        eigvals = sv**2
        explained_var = eigvals / eigvals.sum()
        cum_explained_var = np.cumsum(explained_var)
        n_dims_target = int(np.argmax(cum_explained_var >= var_target_mca) + 1)

        c1, c2 = st.columns(2)
        with c1:
            fig_mca_var, ax = plt.subplots(figsize=(6, 4))
            ax.plot(range(1, len(cum_explained_var) + 1), cum_explained_var, marker="o", linestyle="--")
            ax.axhline(y=var_target_mca)
            ax.set_xlabel("Dimensiones MCA")
            ax.set_ylabel("Varianza acumulada explicada")
            ax.set_title("MCA - Varianza acumulada")
            ax.grid(True)
            st.pyplot(fig_mca_var)
            st.write(f"Dimensiones para ≥ {var_target_mca*100:.0f}%: **{n_dims_target}**")

        with c2:
            coords = mca_cirrosis.fs_r(N=3)
            y_train_align = y_train.iloc[:coords.shape[0]]
            fig_mca_sc, ax2 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=y_train_align, alpha=0.7, ax=ax2)
            ax2.set_xlabel("Dimensión 1")
            ax2.set_ylabel("Dimensión 2")
            ax2.set_title("MCA: Dim 1 vs Dim 2")
            ax2.legend(title="Clase")
            st.pyplot(fig_mca_sc)

        # Contribución de variables (Top-K)
        loadings_cat = pd.DataFrame(mca_cirrosis.fs_c()[:, :n_dims_target], index=X_train_encoded.columns)
        contrib = (loadings_cat**2).div((loadings_cat**2).sum(axis=0), axis=1)
        contrib_total = contrib.sum(axis=1).sort_values(ascending=False)
        top_contrib = contrib_total.head(top_k_mca)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f"**Top-{top_k_mca} aportes (MCA)** — sobre {n_dims_target} dim")
            fig_mca_bar, ax3 = plt.subplots(figsize=(8, 4))
            top_contrib.plot(kind="bar", ax=ax3)
            ax3.set_ylabel("Contribución total")
            ax3.set_title("Aporte de variables/dummies (MCA)")
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
            fig_mca_bar.tight_layout()
            st.pyplot(fig_mca_bar)

        with c4:
            st.markdown("**Resumen (MCA)**")
            st.markdown(
                f"- Varianza objetivo: **{var_target_mca*100:.0f}%**  \n"
                f"- Dimensiones usadas: **{n_dims_target}**  \n"
            )
else:
    st.info("Selecciona variables categóricas para ejecutar MCA.")

# -----------------------------------------------------
# 2.2. PCA  (filtros dentro de la subsección)
# -----------------------------------------------------
st.markdown("## 2.2. PCA")

df_num = df.select_dtypes(include=["int64", "float64"]).copy()

# Controles PCA (en línea, no sidebar)
pca_ctrl = st.container()
with pca_ctrl:
    c_p1, c_p2 = st.columns([1, 1])
    var_target_pca = c_p1.slider(
        "Varianza objetivo (%)",
        min_value=80, max_value=99, value=80, step=1, key="pca_var_pct"
    ) / 100.0
    top_k_pca = c_p2.slider(
        "Top-K variables",
        min_value=5, max_value=50, value=15, step=1, key="pca_topk"
    )

if df_num.empty:
    st.warning("No hay variables numéricas para PCA.")
else:
    y = df["Stage"]
    X_train, X_test, y_train, y_test = train_test_split(
        df_num, y, stratify=y, test_size=0.33, random_state=1
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # PCA completo (para varianza acumulada y PC1 vs PC2)
    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_scaled)
    explained_cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_pc_target = int(np.argmax(explained_cum >= var_target_pca) + 1)

    c5, c6 = st.columns(2)
    with c5:
        fig_pca_var, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(explained_cum) + 1), explained_cum, marker="o", linestyle="--")
        ax.axhline(y=var_target_pca)
        ax.set_xlabel("Número de componentes principales")
        ax.set_ylabel("Varianza acumulada explicada")
        ax.set_title("PCA - Varianza acumulada")
        ax.grid(True)
        st.pyplot(fig_pca_var)
        st.write(f"Componentes para ≥ {var_target_pca*100:.0f}%: **{n_pc_target}**")

    with c6:
        fig_pca_sc, ax2 = plt.subplots(figsize=(6, 4))
        y_train_align = y_train.iloc[:X_pca_full.shape[0]]
        sns.scatterplot(x=X_pca_full[:, 0], y=X_pca_full[:, 1], hue=y_train_align, alpha=0.7, ax=ax2)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title("PCA: PC1 vs PC2")
        ax2.legend(title="Clase")
        st.pyplot(fig_pca_sc)

    # Loadings y Top-K por PCs seleccionadas
    loadings = pd.DataFrame(
        pca_full.components_.T,
        columns=[f"PC{i+1}" for i in range(pca_full.n_components_)],
        index=X_train.columns
    )
    var_importance = (loadings.iloc[:, :n_pc_target] ** 2).sum(axis=1).sort_values(ascending=False)
    top_vars_pca = var_importance.head(top_k_pca)

    c7, c8 = st.columns(2)
    with c7:
        st.markdown(f"**Top-{top_k_pca} variables PCA** — sobre {n_pc_target} PCs")
        fig_pca_bar, ax3 = plt.subplots(figsize=(8, 4))
        top_vars_pca.plot(kind="bar", ax=ax3)
        ax3.set_ylabel("Aporte total (suma de cuadrados de loadings)")
        ax3.set_title("Aporte por variable a PCs seleccionadas")
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
        fig_pca_bar.tight_layout()
        st.pyplot(fig_pca_bar)

    with c8:
        # Resumen con PCA a varianza objetivo (para confirmar números)
        pca_target = PCA(n_components=var_target_pca)
        _ = pca_target.fit_transform(X_scaled)
        st.markdown("**Resumen (PCA)**")
        st.markdown(
            f"- Varianza objetivo: **{var_target_pca*100:.0f}%**  \n"
            f"- Componentes usadas: **{pca_target.n_components_}**  \n"
            f"- Varianza acumulada lograda: **{pca_target.explained_variance_ratio_.sum()*100:.2f}%**"
        )

# ______________________________________________________________________________________________________
# ______________________________________________________________________________________________________
st.markdown("## 2.3. Concatenar las dos matrices")

# ========= 0) Preparación: columnas y split común por índices =========
y_full = df["Stage"]
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Asegura que 'Stage' no esté en cat_cols
cat_cols = [c for c in cat_cols if c != "Stage"]

# Split por índices para alinear filas entre numéricas y categóricas
idx_train, idx_test = train_test_split(
    df.index, stratify=y_full, test_size=0.33, random_state=1
)
y_train = y_full.loc[idx_train]
y_test  = y_full.loc[idx_test]

X_num_train = df.loc[idx_train, num_cols]
X_num_test  = df.loc[idx_test,  num_cols]

X_cat_train = df.loc[idx_train, cat_cols]
X_cat_test  = df.loc[idx_test,  cat_cols]

# ========= 1) PCA sobre numéricas =========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_num_train)  # fit solo con train
X_test_scaled  = scaler.transform(X_num_test)       # transform en test

pca = PCA(n_components=8)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

# ========= 2) MCA sobre categóricas con columnas alineadas =========
# Dummies en train y test, y alineación de columnas
X_train_encoded = pd.get_dummies(X_cat_train, drop_first=False)
X_test_encoded  = pd.get_dummies(X_cat_test,  drop_first=False)

# Alinear columnas de test a las de train (rellenar faltantes con 0)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Modelo prince.MCA (evita sombrear el import `mca`)
mca_pr = prince.MCA(n_components=6, random_state=42)
mca_pr = mca_pr.fit(X_train_encoded)

X_train_mca = mca_pr.transform(X_train_encoded)
X_test_mca  = mca_pr.transform(X_test_encoded)

# ========= 3) DataFrames con índices y nombres de columnas =========
X_train_pca_df = pd.DataFrame(
    X_train_pca, index=idx_train, columns=[f"PCA_{i+1}" for i in range(X_train_pca.shape[1])]
)
X_test_pca_df = pd.DataFrame(
    X_test_pca, index=idx_test, columns=[f"PCA_{i+1}" for i in range(X_test_pca.shape[1])]
)

# prince devuelve DataFrame; asegurar índices correctos y nombres
X_train_mca_df = pd.DataFrame(
    X_train_mca.values, index=idx_train, columns=[f"MCA_{i+1}" for i in range(X_train_mca.shape[1])]
)
X_test_mca_df = pd.DataFrame(
    X_test_mca.values, index=idx_test, columns=[f"MCA_{i+1}" for i in range(X_test_mca.shape[1])]
)

# ========= 4) Concatenación final =========
X_train_final = pd.concat([X_train_pca_df, X_train_mca_df], axis=1)
X_test_final  = pd.concat([X_test_pca_df,  X_test_mca_df],  axis=1)

# ========= 5) Vista rápida =========
cA, cB = st.columns(2)
with cA:
    st.subheader("Train: PCA+MCA (shape)")
    st.write(X_train_final.shape)
    st.dataframe(X_train_final.head(10))
with cB:
    st.subheader("Test: PCA+MCA (shape)")
    st.write(X_test_final.shape)
    st.dataframe(X_test_final.head(10))
# ______________________________________________________________________________________________________

# __________________________________________________________________________________________________


st.markdown("""## 2.4. Modelado""")


# --- Filtro único de la subsección (por defecto: Logistic Regression)
model_name_24 = st.selectbox(
    "Elige el modelo a evaluar (CV 5-fold)",
    options=[
        "Logistic Regression", "KNN", "SVC", 
        "Decision Tree", "Random Forest", 
        "ExtraTrees", "HistGradientBoosting"
    ],
    index=0,
    key="model_sel_24"
)

# --- Construcción del modelo según selección
def build_model(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(
            multi_class="multinomial", 
            solver="lbfgs", 
            max_iter=2000, 
            class_weight="balanced", 
            random_state=42
        )
    if name == "KNN":
        return KNeighborsClassifier(random_state=42)
    if name == "SVC":
        return SVC(random_state=42)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    if name == "Random Forest":
        return RandomForestClassifier(random_state=42)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(random_state=42)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(random_state=42)
    
    raise ValueError("Modelo no soportado")
    
modelo_24 = build_model(model_name_24)

# --- CV estratificado para mayor estabilidad
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(modelo_24, X_train_final, y_train, cv=cv5, scoring="accuracy", n_jobs=-1)

st.subheader("Resultados de validación cruzada")
st.write(f"*Modelo:* {model_name_24}")
st.write(f"*Accuracy (media CV):* {scores.mean():.4f}  |  *Std:* {scores.std():.4f}")

# __________________________________
st.markdown("""## 2.5. Ajuste de hiperparámetros""")

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform

# --- Filtro único de la subsección (por defecto: Logistic Regression)

model_name_25 = st.selectbox(
    "Elige el modelo a ajustar (RandomizedSearchCV)",
    options=[
        "Logistic Regression", "KNN", "SVC", 
        "Decision Tree", "Random Forest",
        "ExtraTrees", "HistGradientBoosting" 
    ],
    index=0,
    key="model_sel_25"
)
# --- Espacios de búsqueda por modelo (evitando combinaciones inválidas)
def get_model_and_searchspace(name: str):
    if name == "Logistic Regression":
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs", penalty="l2", max_iter=5000, class_weight="balanced", random_state=42)
        # solo C para evitar incompatibilidades
        param_dist = {"C": loguniform(1e-3, 1e2)}
        return model, param_dist, "accuracy"
    if name == "KNN":
        model = KNeighborsClassifier()
        param_dist = {
            "n_neighbors": randint(3, 30),
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
        }
        return model, param_dist, "accuracy"
    if name == "SVC":
        model = SVC(probability=False, random_state=42)
        # espacio mixto simple; si kernel='linear', gamma se ignora; no pasa nada en SVC
        param_dist = {
            "C": loguniform(1e-2, 1e2),
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"],
            "degree": randint(2, 5),  # aplica si poly
        }
        return model, param_dist, "accuracy"
    if name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
        param_dist = {
            "max_depth": randint(3, 20),
            "min_samples_split": randint(2, 10),
            #"min_samples_leaf": randint(1, 20),
            "criterion": ["gini", "entropy"],
        }
        return model, param_dist, "accuracy"
    if name == "Random Forest":
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_dist = {
            "n_estimators": randint(100, 600),
            "max_depth": randint(3, 40),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 20),
            "max_features": ["sqrt", "log2", None],
        }
        return model, param_dist, "accuracy"
    if name == "ExtraTrees":
        model = ExtraTreesClassifier(random_state=42, n_jobs=-1)
        param_dist = {
            "n_estimators": randint(50, 200),
            "max_depth": randint(5, 30),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 20),
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        }
        return model, param_dist, "accuracy"
    if name == "HistGradientBoosting":
        model = HistGradientBoostingClassifier(random_state=42)
        param_dist = {
            "max_iter": randint(50, 200),
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": randint(2, 10),
            "max_leaf_nodes": randint(10, 50),
        }
        return model, param_dist, "accuracy"
    raise ValueError("Modelo no soportado")

estimator_25, searchspace_25, metric_25 = get_model_and_searchspace(model_name_25)

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    estimator=estimator_25,
    param_distributions=searchspace_25,
    n_iter=25,
    cv=cv5,
    scoring=metric_25,
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search.fit(X_train_final, y_train)

st.subheader("Mejores hiperparámetros")
st.write(f"*Modelo:* {model_name_25}")
st.write("*Best params:*", random_search.best_params_)
st.write(f"*Mejor {metric_25} (CV):* {random_search.best_score_:.4f}")

# Guardar el mejor estimador en session_state para reusarlo en 2.6
if "best_estimators" not in st.session_state:
    st.session_state.best_estimators = {}
st.session_state.best_estimators[model_name_25] = random_search.best_estimator_

# __________________________________
st.markdown("""## 2.6. Comparación de modelos optimizados""")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Filtro único de la subsección (por defecto: Logistic Regression)
model_name_26 = st.selectbox(
    "Elige el modelo a evaluar en Test",
    options=["Logistic Regression", "KNN", "SVC", "Decision Tree", "Random Forest", "ExtraTrees", "HistGradientBoosting"],
    index=0,
    key="model_sel_26"
)

# Recuperar el mejor estimador si ya fue ajustado en 2.5; si no, construir y entrenar rápido
if "best_estimators" in st.session_state and model_name_26 in st.session_state.best_estimators:
    modelo_26 = st.session_state.best_estimators[model_name_26]
else:
    # fallback: usa el modelo por defecto sin tuning
    modelo_26 = build_model(model_name_26)
    # (opcional) podrías entrenar sobre todo el train antes de evaluar
    # pero la comparación se verá mejor si haces CV o tuning. Aquí entrenamos simple:
    modelo_26.fit(X_train_final, y_train)

# CV del modelo final (opcional para mostrar referencia)
scores_cv = cross_val_score(modelo_26, X_train_final, y_train, cv=cv5, scoring="accuracy", n_jobs=-1)
mean_cv = scores_cv.mean()
std_cv = scores_cv.std()

# Entrenamiento en train y evaluación en test
modelo_26.fit(X_train_final, y_train)
y_pred = modelo_26.predict(X_test_final)
acc_test = accuracy_score(y_test, y_pred)

st.markdown(f"### 📌 Modelo: {model_name_26}")
st.markdown(f"*Accuracy CV (media ± std):* {mean_cv:.4f} ± {std_cv:.4f}")
st.markdown(f"*Accuracy Test:* {acc_test:.4f}")
st.text("📋 Classification Report (Test):")
st.text(classification_report(y_test, y_pred))
st.text("🧩 Matriz de Confusión (Test):")
st.write(pd.DataFrame(confusion_matrix(y_test, y_pred), index=sorted(y_test.unique()), columns=sorted(y_test.unique())))

# Definir resultados fijos manualmente
resultados = {
    "Modelo": [
        "Random Forest",
        "KNN",
        "HistGradientBoosting",
        "ExtraTrees",
        "Decision Tree",
        "SVM",
        "Logistic Regression"
    ],
    "Accuracy CV (media)": [0.9036, 0.9041, 0.8952, 0.8935, 0.8539, 0.7861, 0.5688],
    "Accuracy CV (std)":   [0.0035, 0.0032, 0.0041, 0.0035, 0.0062, 0.0031, 0.0053],
    "Accuracy Test":       [0.9103, 0.9093, 0.8952, 0.9022, 0.8641, 0.7960, 0.5748]
}

# Crear DataFrame
df_resultados = pd.DataFrame(resultados)

# Ordenar por Accuracy Test (descendente)
df_resultados = df_resultados.sort_values(by="Accuracy Test", ascending=False)

# Mostrar en Streamlit
st.markdown("### 📊 Resumen Comparativo de Modelos")
st.dataframe(df_resultados.style.format(precision=4), use_container_width=True)



# Párrafo de análisis
analisis = """
**Análisis**  
Tras evaluar diversos modelos de clasificación con sus respectivos hiperparámetros optimizados, se concluye que los algoritmos con mejor desempeño en términos de *accuracy* son *Random Forest (0.9128)* y *KNN (0.9110)*.

En contraste, el modelo de *Regresión Logística (Softmax: 0.5756)* muestra el rendimiento más bajo, lo que indica que no es el más adecuado para la tarea de clasificación del estadío de la cirrosis.

Por otro lado, los modelos *HistGradientBoosting* y *ExtraTrees* presentan resultados similares, siendo el primero ligeramente superior (accuracy media de 0.8919 frente a 0.8916); ambos superan a *Decision Tree* y *SVM*, aunque no alcanzan el rendimiento de *Random Forest* ni de *KNN*.

Con base en estos resultados, se determina que **Random Forest** es el modelo más adecuado para utilizar con el conjunto de datos generado mediante **PCA** y **MCA**.
"""

# Mostrar el análisis en Streamlit
st.markdown(analisis)

# === FIN SECCIÓN 2 ===








# ________________________________________________________________________________________________________________________________________________________________


st.markdown("""# 3. RFE""")

# Convertir Stage a categórica
df["Stage"] = df["Stage"].astype("category")

# Definir variables categóricas y numéricas
categorical = df.select_dtypes(include=["object","category"])
categorical_features = categorical.columns.drop("Stage").tolist()
numerical_features = df.select_dtypes(include=["int64","float64"]).columns.tolist()

# Separar X e y
X = df[categorical_features + numerical_features]
y = df["Stage"]

# Partición train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

# OneHotEncoder compatible con distintas versiones de sklearn
try:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse=False)

# Preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OHE, categorical_features),
    ]
)

# Modelos disponibles
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(random_state=42),
}

# ---- Control: seleccionar 1 modelo (por defecto Random Forest) ----
model_names = list(models.keys())
default_index = model_names.index("Decision Tree") if "Random Forest" in model_names else 0
modelo_elegido = st.selectbox("Modelo a ejecutar", options=model_names, index=default_index, key="rfe_modelo")
model = models[modelo_elegido]

st.title("Resultados de Selección de Características con RFE-CV")
st.subheader(f"Modelo: {modelo_elegido}")

# RFECV
rfe = RFECV(
    estimator=model,
    step=1,
    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42),
    scoring="accuracy",
    n_jobs=-1,
)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selection", rfe),
    ("model", model),
])

# Entrenar
pipeline.fit(X_train, y_train)

# Evaluar
accuracy_test = pipeline.score(X_test, y_test)
mask = pipeline.named_steps["feature_selection"].support_
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
selected_names = feature_names[mask]

# Mostrar en la app
st.write(f"Accuracy en test set: {accuracy_test:.3f}")
st.write(f"Variables seleccionadas: {len(selected_names)}")
st.write(f"Nombres: {list(selected_names)}")


# =========================
# TABLA RESUMEN
# =========================
st.header("Tabla Comparativa de Modelos")

# Definir manualmente los resultados proporcionados
resultados = {
    "Modelo": ["Decision Tree", "Random Forest", "Logistic Regression","Extra Trees"],
    "Accuracy": [0.922, 0.946, 0.553, 0.948],
    "N° de Variables": [3, 6, 25, 7],
    "Variables Seleccionadas": [
        ["num_N_Days", "numAlbumin", "num_Prothrombin"],
        ["num_N_Days", "numAge", "numBilirubin", "numAlbumin", "numPlatelets", "num_Prothrombin"],
        ["num_N_Days", "numBilirubin", "numCholesterol", "numAlbumin", "num_Copper",
         "num_SGOT", "numTryglicerides", "numPlatelets", "num_Prothrombin",
         "cat_Status_C", "catStatus_CL", "catStatus_D", "cat_Drug_D-penicillamine",
         "cat_Drug_Placebo", "catSex_F", "catSex_M", "catAscites_N", "cat_Ascites_Y",
         "cat_Hepatomegaly_N", "catHepatomegaly_Y", "catSpiders_N", "cat_Spiders_Y",
         "cat_Edema_N", "catEdema_S", "cat_Edema_Y"],
        ["num_N_Days", "numAge", "numBilirubin", "numAlbumin", "numCopper", "num_Platelets", 
         "num__Prothrombin"]
    ]
}

# Convertir a DataFrame
df_resultados = pd.DataFrame(resultados)

# Mostrar tabla en formato científico
st.dataframe(df_resultados.style.format(precision=3))


st.subheader("Análisis RFECV")

st.markdown("""
Se aplicó el método RFE (eliminación recursiva de características) con validación cruzada (CV) para la selección de variables, entrenando cuatro modelos: árboles de decisión, random forest, regresión logística 
y extra trees. Los resultados mostraron un buen desempeño en los modelos Extra Trees (0.948), Random Forest (0.946) y Decision Tree (0.922). Durante el proceso, cada modelo fue reduciendo progresivamente las 
variables hasta conservar 7, 6 y 3 características respectivamente, siendo este el enfoque que permitió la mayor reducción de características.
""")


st.header("Conclusión final:")

st.markdown("""
En el proceso de reducción de características se observó que, para las variables categóricas, la aplicación de Chi² e información mutua (posterior al OneHotEncoder) permitió reducirlas a 9. En el caso de las variables numéricas, los métodos ANOVA e información mutua no resultaron efectivos, ya que la reducción afectaba negativamente el desempeño de los modelos. Al combinar ambas selecciones (20 variables en total) y entrenar los modelos, el mejor rendimiento se obtuvo con HistGradientBoosting, alcanzando un accuracy de 0.9582.
Por otro lado, mediante MCA y PCA se logró una reducción a 6 dimensiones y 8 componentes principales (14 características en total). Con este conjunto, los modelos con mejor desempeño fueron Random Forest (0.9036) y KNN (0.9041), ambos superando el 90% de accuracy.
Finalmente, con la eliminación recursiva de características (RFECV) se consiguió la mayor reducción con el modelo Decision Tree, que conservó solo 3 variables con un accuracy de 0.922. Sin embargo, los modelos Random Forest (0.946) y Extra Trees (0.948) mostraron un mejor desempeño, aunque utilizando una mayor cantidad de características (6 y 7 respectivamente). Cabe destacar que el modelo HistGradientBoosting no fue entrenado porque no es compatible con esta técnica de selección.

**HistGradientBoosting** se posiciona como el **mejor modelo** para la clasificación de etapas de cirrosis, al lograr el **mayor accuracy** tanto en validación cruzada (~96 %) como en prueba (~95–96 %), con una **brecha mínima** entre ambos que sugiere muy buena generalización.
""")




