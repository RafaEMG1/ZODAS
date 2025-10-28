# zodas_map.py
from pathlib import Path
import json
import folium
from pyproj import Transformer

def _to_int(x):
    try:
        s = "".join(ch for ch in str(x).strip() if ch.isdigit())
        return int(s) if s else None
    except:
        return None

def esri_to_geojson_reproject(esri_path: Path):
    esri = json.loads(esri_path.read_text(encoding="utf-8", errors="ignore"))
    feats_esri = esri.get("features", [])
    wkid = (esri.get("spatialReference") or {}).get("latestWkid") or (esri.get("spatialReference") or {}).get("wkid")
    wkid = int(wkid) if wkid is not None else 4326

    if wkid != 4326:
        transformer = Transformer.from_crs(f"EPSG:{wkid}", "EPSG:4326", always_xy=True)
        def reproj_ring(r): return [list(transformer.transform(x,y)) for x,y in r]
    else:
        def reproj_ring(r): return r

    gj_features = []
    for f in feats_esri:
        props = f.get("properties") or f.get("attributes") or {}
        mp = props.get("MpCodigo") or props.get("MPCodigo") or props.get("mpcodigo")
        props["MpCodigo_norm"] = _to_int(mp)
        props["MpNombre"] = props.get("MpNombre") or props.get("MPNombre") or props.get("mpnombre") or ""
        geom = f.get("geometry") or {}
        if "rings" in geom:
            rings = geom["rings"]
            rings_ll = [reproj_ring(r) for r in rings]
            geom_gj = {"type": "Polygon", "coordinates": rings_ll}
        else:
            continue
        gj_features.append({"type": "Feature", "properties": props, "geometry": geom_gj})

    return {"type": "FeatureCollection", "features": gj_features}

def build_map(geojson_fc, df_geo, df_asigs, df_centros, dane_to_name, dane_to_xy):
    # centro del mapa
    coords_all = [(float(r.lat), float(r.lon)) for _, r in df_geo.dropna(subset=["lat","lon"]).iterrows()]
    clat = sum(a for a,_ in coords_all)/len(coords_all) if coords_all else 2.93
    clon = sum(b for _,b in coords_all)/len(coords_all) if coords_all else -75.28
    fmap = folium.Map(location=[clat, clon], zoom_start=8, tiles="cartodbpositron")

    # paleta
    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    centros_orden = sorted(df_centros["centro_dane"].dropna().astype(int).unique()) if not df_centros.empty else []
    color_by_centro = {int(c): palette[i % len(palette)] for i, c in enumerate(centros_orden)}

    # mapping municipios->centro
    cluster_by_mun = {}
    if not df_asigs.empty:
        for _, r in df_asigs.iterrows():
            cluster_by_mun[int(r["municipio_dane"])] = int(r["centro_dane"])

    # capa polígonos
    def style_fn(feat):
        dane = feat["properties"].get("MpCodigo_norm")
        if dane in cluster_by_mun:
            c = cluster_by_mun[dane]
            return {"fillColor": color_by_centro.get(int(c), "#cccccc"), "color": "#555", "weight": 0.8, "fillOpacity": 0.6}
        else:
            return {"fillColor": "#ddd", "color": "#999", "weight": 0.5, "fillOpacity": 0.3}

    folium.GeoJson(
        geojson_fc,
        name="Clúster por polígono",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=["MpNombre","MpCodigo_norm"], aliases=["Municipio","DANE"])
    ).add_to(fmap)

    # puntos de municipios
    for _, r in df_geo.dropna(subset=["lat","lon"]).iterrows():
        dane = int(r["dane_code"])
        lat, lon = float(r["lat"]), float(r["lon"])
        cc = cluster_by_mun.get(dane)
        color = color_by_centro.get(int(cc), "#444444") if cc is not None else "#aaaaaa"
        folium.CircleMarker(
            location=[lat,lon], radius=3, color=color, fill=True, fill_opacity=0.85,
            popup=f'{r["municipio"]} (DANE {dane})' + (f' → Centro {dane_to_name.get(int(cc), "")} ({int(cc)})' if cc else "")
        ).add_to(fmap)

    # marcadores de centros
    if not df_centros.empty:
        for _, r in df_centros.iterrows():
            c_dane = int(r["centro_dane"])
            lat, lon = dane_to_xy.get(c_dane, (None, None))
            if None in (lat,lon): continue
            folium.CircleMarker(
                location=[lat,lon], radius=7, color=color_by_centro.get(c_dane, "#333"), fill=True, fill_opacity=0.95,
                popup=f'Centro: {dane_to_name.get(c_dane, c_dane)} (DANE {c_dane})'
            ).add_to(fmap)

    return fmap
