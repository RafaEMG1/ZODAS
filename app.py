# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from streamlit_folium import st_folium
import altair as alt

from zodas_core import (
    load_huila_data, diagnostics, build_and_solve, extract_results,
    to_dane
)
from zodas_map import esri_to_geojson_reproject, build_map

st.set_page_config(page_title="ZODAS - Optimización de Clústeres", layout="wide")

st.title("ZODAS – Optimización de Clústeres Agrícolas (Huila)")

with st.sidebar:
    st.header("Parámetros de entrada")

    datos = load_huila_data()
    # Productos disponibles que existan en Oferta y Demanda
    productos_validos = [p for p in (datos["pact"] or datos["productos"]) if p in datos["oferta"][datos["I_ids"][0]] and p in datos["demanda"][datos["J_ids"][0]]]
    producto = st.selectbox("Producto", productos_validos, index=0)

    dmax = st.number_input("Distancia máxima O→A (km)", min_value=1.0, max_value=5000.0, value=150.0, step=1.0)
    p = st.number_input("Cantidad de clústeres (p)", min_value=1, max_value=max(1,len(datos["K_ids"])), value=int(datos["p_default"]), step=1)

    # centroides forzados (por nombre)
    dane_to_name = datos["dane_to_name"]
    name_to_dane = {v: k for k, v in dane_to_name.items()}

    # candidatos: todos los posibles K (mapeados a DANE y nombre)
    candidatos = sorted({dane_to_name.get(to_dane(k, datos["id_to_dane"], dane_to_name), f"ID {k}") for k in datos["K_ids"]})
    forced_names = st.multiselect("Municipios que deben ser centros (mínimo 1)", options=candidatos, default=candidatos[:1])
    forced_danes = [name_to_dane[n] for n in forced_names if n in name_to_dane]

    run_btn = st.button("Resolver")

# mensajes previos de infactibilidad
prev_reasons = diagnostics(
    datos["oferta"], datos["demanda"], datos["cap"], datos["K_ids"],
    producto, int(p), float(dmax), datos["distOA"], datos["I_ids"], datos["J_ids"],
    datos["id_to_dane"], datos["dane_to_name"]
)

if prev_reasons:
    st.warning("**Chequeos previos** (posibles causas de infactibilidad):\n- " + "\n- ".join(prev_reasons))

if len(forced_danes) < 1:
    st.info("Selecciona **al menos un municipio centroide** en la barra lateral para continuar.")

if run_btn and len(forced_danes) >= 1:
    with st.spinner("Resolviendo modelo..."):
        m, msgs, forced_k = build_and_solve(producto, int(p), float(dmax), forced_danes, datos)

    if m is None:
        st.error("No se pudo construir/solucionar el modelo: " + "; ".join(msgs))
        st.stop()

    if msgs:
        for msg in msgs:
            st.warning(msg)

    # resultados
    df_centros, df_asigs, df_X, df_Y, obj_val = extract_results(m, datos, producto)

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.metric("Costo total", f"{obj_val:,.2f}" if obj_val is not None else "—")
        st.metric("Centros abiertos", len(df_centros))
    with col2:
        st.metric("Arcos O→A con flujo", len(df_X))
    with col3:
        st.metric("Arcos A→D con flujo", len(df_Y))

    st.subheader("Centroides abiertos")
    st.dataframe(df_centros, use_container_width=True)

    st.subheader("Asignaciones municipio → centro")
    st.dataframe(df_asigs, use_container_width=True)

    with st.expander("Flujos O→A (X)"):
        st.dataframe(df_X, use_container_width=True)
    with st.expander("Flujos A→D (Y)"):
        st.dataframe(df_Y, use_container_width=True)

    # Descargas
    st.download_button("Descargar centroides (CSV)", df_centros.to_csv(index=False).encode("utf-8"), "result_centroides.csv", "text/csv")
    st.download_button("Descargar asignaciones (CSV)", df_asigs.to_csv(index=False).encode("utf-8"), "result_asignaciones.csv", "text/csv")
    st.download_button("Descargar flujos O→A (CSV)", df_X.to_csv(index=False).encode("utf-8"), "result_flujos_X.csv", "text/csv")
    st.download_button("Descargar flujos A→D (CSV)", df_Y.to_csv(index=False).encode("utf-8"), "result_flujos_Y.csv", "text/csv")

    # Mapa
    st.subheader("Mapa de clústeres")
    esri_path = Path(__file__).parent / "data" / "huila_municipios.json"
    try:
        geo_fc = esri_to_geojson_reproject(esri_path)
        from streamlit.components.v1 import html

        fmap = build_map(geo_fc, datos["df_geo"], df_asigs, df_centros, datos["dane_to_name"], datos["dane_to_xy"])
        
        # Renderizamos el HTML del mapa directamente (evita la serialización JSON de funciones)
        map_html = fmap.get_root().render()
        html(map_html, height=650)   # puedes ajustar el alto
    except Exception as e:
        st.error(f"No se pudo generar el mapa: {e}")

    # Curva costo vs p (opcional)
    st.subheader("Curva costo vs cantidad de clústeres (opcional)")
    with st.form("curve_form"):
        p_min = st.number_input("p mínimo", 1, max(1,len(datos["K_ids"])), max(1, int(p)-2))
        p_max = st.number_input("p máximo", 1, max(1,len(datos["K_ids"])), max(int(p), int(p)+2))
        run_curve = st.form_submit_button("Calcular curva")
    if run_curve:
        from zodas_core import cost_curve
        p_list = list(range(int(p_min), int(p_max)+1))
        df_curve = cost_curve(p_list, producto, float(dmax), forced_danes, datos)
        st.dataframe(df_curve, use_container_width=True)
        chart = alt.Chart(df_curve.dropna()).mark_line(point=True).encode(
            x=alt.X('p:Q', title='# de clústeres'),
            y=alt.Y('costo:Q', title='Costo total'),
            tooltip=['p','costo','status']
        )
        st.altair_chart(chart, use_container_width=True)
