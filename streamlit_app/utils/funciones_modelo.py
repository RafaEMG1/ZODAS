import pydeck as pdk
import pandas as pd
import numpy as np

def color_palette(n):
    rng = np.random.default_rng(42)
    colors = (rng.uniform(50, 200, size=(n, 3))).astype(int).tolist()
    return colors

def mapa_clusters(df_assign):
    # df_assign: municipio, lat, lon, centroide, centro_lat, centro_lon
    centros = df_assign[["centroide","centro_lat","centro_lon"]].drop_duplicates().reset_index(drop=True)
    centros["cluster_id"] = range(len(centros))

    df_plot = df_assign.merge(centros, on="centroide", how="left")
    palette = color_palette(len(centros))
    df_plot["color"] = df_plot["cluster_id"].apply(lambda i: palette[int(i)])

    center_lat = df_plot["lat"].mean()
    center_lon = df_plot["lon"].mean()

    layer_points = pdk.Layer(
        "ScatterplotLayer",
        data=df_plot,
        get_position='[lon, lat]',
        get_fill_color='color',
        get_radius=1800,
        pickable=True,
    )
    layer_centers = pdk.Layer(
        "ScatterplotLayer",
        data=centros.rename(columns={"centro_lat":"lat","centro_lon":"lon"}),
        get_position='[lon, lat]',
        get_fill_color=[0, 0, 0],
        get_radius=3000,
        pickable=True,
    )
    tooltip = {"html": "<b>{municipio}</b><br/>Centroide: {centroide}<br/>Prod: {produccion}"}
    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=7)
    return pdk.Deck(layers=[layer_points, layer_centers], initial_view_state=view_state, tooltip=tooltip)
