import streamlit as st
import pandas as pd

from utils.funciones_modelo import solve_p_median, cost_vs_p
from utils.mapas import mapa_clusters

st.set_page_config(page_title="ZODAS – Optimización de Clústeres (MIP)", layout="wide")
st.title("Optimización de Clústeres Logísticos – ZODAS (MIP p-mediana)")

st.sidebar.header("Parámetros de entrada")

uploaded = st.sidebar.file_uploader("Sube tu CSV (producción municipal)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    st.sidebar.info("Usando datos DEMO. Reemplace por datos reales para análisis.")
    df = pd.read_csv("data/produccion_municipal_demo.csv")

expected_cols = {"departamento","municipio","lat","lon","producto","produccion"}
if not expected_cols.issubset(set(df.columns)):
    st.error(f"CSV inválido. Debe contener columnas {sorted(expected_cols)}")
    st.stop()

dept = st.sidebar.selectbox("Departamento", options=sorted(df["departamento"].unique()))
df_d = df[df["departamento"] == dept].copy()

prod = st.sidebar.selectbox("Producto", options=sorted(df_d["producto"].unique()))
df_dp = df_d[df_d["producto"] == prod].copy()

R = st.sidebar.number_input("Radio máximo R de asignación (km)", min_value=5.0, max_value=400.0, value=120.0, step=5.0)
S = st.sidebar.number_input("Separación mínima S entre centros (km)", min_value=0.0, max_value=300.0, value=10.0, step=5.0)

municipios = sorted(df_dp["municipio"].unique())
p_default = min(3, len(municipios))
p = st.sidebar.slider("Número de clústeres (p)", min_value=1, max_value=max(1, len(municipios)), value=p_default, step=1)

fixed = st.sidebar.multiselect("Fijar centroides (opcional)", options=municipios,
                               help="Fuerza y_j=1 para los municipios seleccionados.")
st.sidebar.caption("Nota: p ≥ #centroides fijos; y los fijos deben respetar S.")

# Botón principal
if st.sidebar.button("Optimizar (resolver MIP)"):
    with st.spinner("Resolviendo modelo MIP p-mediana..."):
        df_assign, df_centros, cost, warns = solve_p_median(
            df_dp, p=p, R=R, S=S, fixed_centroids=fixed, time_limit=90, msg=False
        )

    if warns:
        for w in warns:
            st.warning(w)

    if df_assign is not None:
        st.subheader("Resultados")

        col1, col2 = st.columns([1,1])
        with col1:
            st.write(f"**Costo total (∑ w_i × d_ij)**: {cost:,.2f}")
            st.dataframe(df_centros.rename(columns={"centro_lat":"lat","centro_lon":"lon"}))

            grupos = df_assign.groupby("centroide")["municipio"].apply(list).reset_index()
            st.write("### Municipios por clúster")
            for _, row in grupos.iterrows():
                st.markdown(f"**{row['centroide']}**: {', '.join(row['municipio'])}")

        with col2:
            deck = mapa_clusters(df_assign)
            st.pydeck_chart(deck, use_container_width=True)

st.divider()
st.subheader("Curva de costo vs número de clústeres (p)")
max_p = min(12, len(municipios))
Ps = list(range(1, max_p+1))
if st.button("Calcular curva (resuelve varios p)"):
    with st.spinner("Calculando..."):
        df_curve = cost_vs_p(df_dp, Ps=Ps, R=R, S=S, fixed_centroids=fixed, time_limit=30)
    st.line_chart(df_curve.set_index("p")["costo"])
    st.dataframe(df_curve)
