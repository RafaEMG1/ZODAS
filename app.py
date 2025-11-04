# ============================================================
# ZODAS – Streamlit (archivo único)
# Optimización de clústeres (Pyomo+HiGHS) + mapa Folium
# - Curva costo vs p sin deadlock (appsi_highs)
# - Tablas SIN códigos DANE
# - Nombres SIN tildes (á→a, ñ→n) y mapa coloreado por DANE
# ============================================================

from pathlib import Path
import re, csv, json, time, gc, unicodedata
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html
import altair as alt

import pyomo.environ as pyo
import folium
from pyproj import Transformer

# ------------------------------------------------------------
# Configuración
# ------------------------------------------------------------
st.set_page_config(page_title="ZODAS - Optimización de Clústeres", layout="wide")
DATA_DIR = Path(__file__).parent / "data"
P_MAX_UI = 20

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def strip_accents(s: str) -> str:
    if s is None: return s
    s = str(s)
    # Normaliza y elimina marcas diacríticas (tildes/ñ→n)
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).replace("ß", "ss")

def _to_int_safe(x):
    if x is None: return None
    s = "".join(ch for ch in str(x).strip() if ch.isdigit())
    return int(s) if s else None

def _v(x):
    try: return pyo.value(x)
    except: 
        try: return float(x)
        except: return x

def _fix_names_in_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.endswith("_nombre") or c in ("centro_nombre","municipio","destino_nombre","origen_nombre")]
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(strip_accents)
    return df

# ------------------------------------------------------------
# Parsers AMPL .dat
# ------------------------------------------------------------
def parse_set(text: str, set_name: str):
    m = re.search(rf"set\s+{set_name}\s*:=\s*(.*?)\s*;", text, re.S | re.I)
    if not m: return []
    return [t.strip() for t in m.group(1).split() if t.strip() and t.strip() != ";"]

def parse_scalar_param(text: str, name: str, cast=float):
    m = re.search(rf"param\s+{name}\s*:=\s*([-\d\.Ee]+)\s*;", text, re.S | re.I)
    return cast(m.group(1)) if m else None

def parse_vector_param(text: str, name: str):
    m = re.search(rf"param\s+{name}\s*:=\s*(.*?)\s*;", text, re.S | re.I)
    out = {}
    if not m: return out
    block = m.group(1)
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        toks = line.split()
        if len(toks) >= 2 and toks[0].isdigit():
            out[int(toks[0])] = float(toks[1])
    return out

def parse_param_table_products(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [l.rstrip() for l in text.splitlines()]
    head_idx = None
    for i, l in enumerate(lines):
        if l.strip().startswith("param"):
            head_idx = i + 1
            break
    assert head_idx is not None, f"Cabecera no encontrada en {path.name}"
    header_line = lines[head_idx].strip()
    header = [h for h in header_line.replace("=", "").split() if not h.endswith(":")]
    data = {}
    for l in lines[head_idx + 1:]:
        s = l.strip()
        if not s or s.startswith("#"): continue
        if s.endswith(";"):
            s = s[:-1]
            if not s: break
        toks = s.split()
        if not toks: continue
        try: i = int(toks[0])
        except: continue
        vals = [float(x) for x in toks[1:1+len(header)]]
        data[i] = dict(zip(header, vals))
    return data, header

def parse_param_matrix(path: Path, name: str):
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    head = None; start = None
    for idx, l in enumerate(lines):
        if l.strip().lower().startswith(f"param {name}".lower()):
            head = [int(x) for x in re.findall(r"(\d+)", lines[idx+1])]
            start = idx + 2
            break
    assert head is not None, f"No se encontró cabecera de {name} en {path.name}"
    mat = {}
    for l in lines[start:]:
        s = l.strip()
        if s.endswith(";"): s = s[:-1]
        if s.endswith("="): continue
        toks = s.split()
        if not toks: continue
        try: row = int(toks[0])
        except: continue
        vals = [float(x) for x in toks[1:len(head)+1]]
        if len(vals) == len(head):
            mat[row] = dict(zip(head, vals))
    rows = sorted(mat.keys())
    return rows, head, mat

# ------------------------------------------------------------
# CSV DANE / nombres / coords
# ------------------------------------------------------------
def load_huila_csv(csv_path: Path) -> pd.DataFrame:
    rows = []
    with open(csv_path, "r", encoding="latin-1", errors="ignore") as f:
        header = next(f).strip().split(",")
        for line in f:
            line = line.rstrip("\r\n")
            if not line: continue
            inner = line
            if inner.startswith('"') and inner.endswith('"'):
                inner = inner[1:-1].replace('""', '"')
            else:
                inner = inner.replace('""', '"')
            row = next(csv.reader([inner], delimiter=",", quotechar='"', doublequote=True))
            rows.append(row)
    df = pd.DataFrame(rows, columns=header)
    df["id_idx"]    = pd.to_numeric(df["id_idx"], errors="coerce").astype("Int64")
    df["dane_code"] = pd.to_numeric(df["dane_code"], errors="coerce").astype("Int64")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["municipio"] = df["municipio"].astype(str).str.strip().map(strip_accents)
    return df[["id_idx","dane_code","municipio","lat","lon"]].dropna(subset=["id_idx","dane_code"])

def build_maps_from_csv(df_geo: pd.DataFrame):
    id_to_dane   = {int(r.id_idx): int(r.dane_code) for _, r in df_geo.iterrows()}
    dane_to_name = {int(r.dane_code): str(r.municipio) for _, r in df_geo.iterrows()}  # ya sin tildes
    dane_to_xy   = {int(r.dane_code): (float(r.lat), float(r.lon)) for _, r in df_geo.iterrows()}
    return id_to_dane, dane_to_name, dane_to_xy

def to_dane(x, id_to_dane, dane_to_name):
    x = int(x)
    if x in id_to_dane: return id_to_dane[x]
    if x in dane_to_name: return x
    return x

# ------------------------------------------------------------
# Carga de datos (cacheado)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_load_data():
    datos_p   = DATA_DIR / "datos.dat"
    oferta_p  = DATA_DIR / "01Oferta.dat"
    demanda_p = DATA_DIR / "02demanda.dat"
    d13_p     = DATA_DIR / "03distancias.dat"
    d24_p     = DATA_DIR / "04distancias.dat"
    csv_p     = DATA_DIR / "huila_geo.csv"

    txt = datos_p.read_text(encoding="utf-8", errors="ignore")
    productos = parse_set(txt, "Productos")
    pact      = parse_set(txt, "ProductosActivos") or productos[:]
    cap       = parse_vector_param(txt, "CapacidadAlmacenamiento")
    p_default = int(parse_scalar_param(txt, "CantidadClusters", cast=float))
    ctr       = parse_scalar_param(txt, "CTr")
    ca        = parse_scalar_param(txt, "CA")
    tonco2    = parse_scalar_param(txt, "TONCO2")

    oferta, headerO = parse_param_table_products(oferta_p)
    demanda, headerD = parse_param_table_products(demanda_p)
    rowsOA, colsOA, distOA = parse_param_matrix(d13_p, "DistanciaOrigenAcopio")
    rowsKD, colsKD, distKD = parse_param_matrix(d24_p, "DistanciaAcopioDestino")

    I_ids = sorted(oferta.keys())
    J_ids = sorted(demanda.keys())
    K_ids = sorted(set(colsOA))

    assert set(I_ids) == set(rowsOA), "Filas DistanciaOrigenAcopio deben coincidir con Oferta (IDs)"
    assert set(J_ids) == set(colsKD), "Columnas DistanciaAcopioDestino deben coincidir con Demanda (IDs)"

    df_geo = load_huila_csv(csv_p)
    id_to_dane, dane_to_name, dane_to_xy = build_maps_from_csv(df_geo)

    prods_oferta  = set.intersection(*[set(d.keys()) for d in oferta.values()]) if oferta else set()
    prods_demanda = set.intersection(*[set(d.keys()) for d in demanda.values()]) if demanda else set()
    productos_validos = sorted(list(prods_oferta & prods_demanda)) or sorted(list((set(pact) | set(productos)) & prods_oferta & prods_demanda))

    # También guardo name_to_dane para joins por nombre
    name_to_dane = {v: k for k, v in dane_to_name.items()}

    return dict(
        productos=productos, pact=pact, cap=cap, p_default=p_default,
        ctr=ctr, ca=ca, tonco2=tonco2,
        oferta=oferta, demanda=demanda, distOA=distOA, distKD=distKD,
        I_ids=I_ids, J_ids=J_ids, K_ids=K_ids,
        df_geo=df_geo, id_to_dane=id_to_dane, dane_to_name=dane_to_name, dane_to_xy=dane_to_xy,
        name_to_dane=name_to_dane,
        productos_validos=productos_validos
    )

# ------------------------------------------------------------
# Diagnósticos (explican infactibilidad)
# ------------------------------------------------------------
def diagnostics(oferta, demanda, cap, K_ids, product, p, dmax, distOA, I_ids, J_ids, id_to_dane, dane_to_name):
    reasons = []
    tot_oferta  = sum(oferta[i].get(product, 0.0) for i in I_ids)
    tot_demanda = sum(demanda[j].get(product, 0.0) for j in J_ids)

    if tot_oferta + 1e-9 < tot_demanda:
        reasons.append(f"Oferta total ({tot_oferta:.2f}) < Demanda total ({tot_demanda:.2f}) para '{product}'.")

    if p > len(K_ids):
        reasons.append(f"p={p} mayor que |K|={len(K_ids)}.")

    caps = [cap.get(k, 0.0) for k in K_ids]
    caps_sorted = sorted(caps, reverse=True)
    cap_max_p = sum(caps_sorted[:p]) if caps_sorted else 0.0
    if cap_max_p + 1e-9 < tot_demanda:
        reasons.append(f"Capacidad máxima con p centros ({cap_max_p:.2f}) < Demanda total ({tot_demanda:.2f}).")

    A_allowed = {(i,k) for i in I_ids for k in K_ids if distOA[i][k] <= float(dmax) + 1e-9}
    unreachable_i = [i for i in I_ids if all((i,k) not in A_allowed for k in K_ids)]
    if unreachable_i:
        names = [strip_accents(dane_to_name.get(to_dane(i,id_to_dane,dane_to_name),'?')) for i in unreachable_i[:10]]
        extra = " ..." if len(unreachable_i) > 10 else ""
        reasons.append(f"{len(unreachable_i)} municipio(s) sin acopio dentro de DMAX={dmax} km: " + ", ".join(names) + extra)

    return reasons

# ------------------------------------------------------------
# Modelo Pyomo + HiGHS
# ------------------------------------------------------------
def build_and_solve(product, p, dmax, forced_danes, datos, backend="highs"):
    oferta  = datos["oferta"]; demanda = datos["demanda"]
    distOA  = datos["distOA"];  distKD  = datos["distKD"]
    cap     = datos["cap"]
    I_ids   = datos["I_ids"];   J_ids   = datos["J_ids"];   K_ids = datos["K_ids"]
    ctr     = datos["ctr"];     ca      = datos["ca"];       tonco2 = datos["tonco2"]
    id_to_dane = datos["id_to_dane"]; dane_to_name = datos["dane_to_name"]

    A_allowed = {(i,k) for i in I_ids for k in K_ids if distOA[i][k] <= float(dmax) + 1e-9}

    def dane_to_k(dane):
        for k in K_ids:
            if to_dane(k, id_to_dane, dane_to_name) == int(dane):
                return k
        return None

    forced_k = [dane_to_k(int(d)) for d in forced_danes]
    forced_k = [k for k in forced_k if k is not None]
    if len(forced_k) > p:
        return None, [f"Seleccionaste {len(forced_k)} centros forzados, que excede p={p}. Reduce el número de centros forzados o aumenta p."], None

    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=I_ids, ordered=True)
    m.K = pyo.Set(initialize=K_ids, ordered=True)
    m.J = pyo.Set(initialize=J_ids, ordered=True)
    m.P = pyo.Set(initialize=[product], ordered=True)
    m.A = pyo.Set(dimen=2, initialize=A_allowed)

    m.X = pyo.Var(m.A, m.P, domain=pyo.NonNegativeReals)
    m.Y = pyo.Var(m.K, m.J, m.P, domain=pyo.NonNegativeReals)
    m.T = pyo.Var(m.K, domain=pyo.Binary)
    m.U = pyo.Var(m.A, domain=pyo.Binary)

    OfertaMax = 1e12

    def obj_rule(_m):
        expr = 0.0
        for (i,k) in _m.A:
            d = distOA[i][k]
            expr += (ctr*d + ca*tonco2*d) * _m.X[(i,k), product]
        for k in _m.K:
            for j in _m.J:
                d = distKD[k][j]
                expr += (ctr*d + ca*tonco2*d) * _m.Y[k, j, product]
        return expr
    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def oferta_rule(_m, i):
        return sum(_m.X[(i,k), product] for k in _m.K if (i,k) in _m.A) <= oferta[i][product]
    m.OfertaR = pyo.Constraint(m.I, rule=oferta_rule)

    def demanda_rule(_m, j):
        return sum(_m.Y[k, j, product] for k in _m.K) == demanda[j][product]
    m.DemandaR = pyo.Constraint(m.J, rule=demanda_rule)

    def balance_rule(_m, k):
        return sum(_m.X[(i,k), product] for i in _m.I if (i,k) in _m.A) == sum(_m.Y[k, j, product] for j in _m.J)
    m.BalanceR = pyo.Constraint(m.K, rule=balance_rule)

    def cap_rule(_m, k):
        return sum(_m.X[(i,k), product] for (i,kk) in _m.A if kk==k) <= cap.get(int(k),0.0) * _m.T[k]
    m.CapR = pyo.Constraint(m.K, rule=cap_rule)

    def act_rule(_m):
        return sum(_m.T[k] for k in _m.K) == p
    m.ActR = pyo.Constraint(rule=act_rule)

    def asig_rule(_m, i):
        return sum(_m.U[(i,k)] for k in _m.K if (i,k) in _m.A) <= 1
    m.AsigR = pyo.Constraint(m.I, rule=asig_rule)

    def link_xu(_m, i, k):
        return sum(_m.X[(i,k), product] for _p in _m.P) <= OfertaMax * _m.U[(i,k)]
    m.LinkXU = pyo.Constraint(m.A, rule=link_xu)

    def link_ut(_m, i, k):
        return _m.U[(i,k)] <= _m.T[k]
    m.LinkUT = pyo.Constraint(m.A, rule=link_ut)

    for k in forced_k:
        m.add_component(f"Force_{k}", pyo.Constraint(expr=m.T[k] == 1))

    try:
        if backend == "appsi_highs":
            solver = pyo.SolverFactory("appsi_highs")
            solver.config.time_limit = 180
            solver.config.load_solution = True
            solver.config.stream_solver = False
        else:
            solver = pyo.SolverFactory("highs")
            solver.options["time_limit"] = 180
            solver.options["mip_rel_gap"] = 0.01
            solver.options["presolve"] = "on"
    except Exception:
        solver = pyo.SolverFactory("appsi_highs")
        solver.config.time_limit = 180
        solver.config.load_solution = True
        solver.config.stream_solver = False

    res = solver.solve(m)
    status = str(res.solver.termination_condition)
    msgs = []
    if status.lower() not in ("optimal", "optimal termination", "feasible"):
        msgs.append("Motivo por el cual es infactible: no se encontró solución factible/óptima. Por favor intente ampliando el número de clústeres (p) o la distancia máxima entre el origen y el acopio (DMAX).")

    return m, msgs, forced_k

# ------------------------------------------------------------
# Resultados (solo nombres, sin DANE) + nombres sin tildes
# ------------------------------------------------------------
def extract_results(m, datos, product):
    id_to_dane = datos["id_to_dane"]; dane_to_name = datos["dane_to_name"]
    distOA = datos["distOA"]; distKD = datos["distKD"]; cap = datos["cap"]
    EPS = 1e-6

    def to_name_from_node(node_id):
        dane = to_dane(node_id, id_to_dane, dane_to_name)
        return strip_accents(dane_to_name.get(int(dane), f"ID {node_id}"))

    # CENTROS
    centros = []
    for k in m.K:
        if _v(m.T[k]) > 0.5:
            fin  = sum(_v(m.X[(i,k), product]) for i in m.I if (i,k) in m.A)
            fout = sum(_v(m.Y[k, j, product]) for j in m.J)
            centros.append({
                "centro_nombre": to_name_from_node(k),
                "flujo_entrante_total": fin,
                "flujo_saliente_total": fout,
                "capacidad": cap.get(int(k), np.nan),
            })
    df_centros = pd.DataFrame(centros).sort_values("centro_nombre")
    df_centros = _fix_names_in_df(df_centros)

    # ASIGNACIONES
    asigs = []
    for i in m.I:
        ks = [k for k in m.K if (i,k) in m.A and _v(m.U[(i,k)]) > 0.5]
        if ks:
            k_best = max(ks, key=lambda kk: _v(m.X[(i,kk), product]))
            fl_i = _v(m.X[(i,k_best), product])
            asigs.append({
                "municipio_nombre": to_name_from_node(i),
                "centro_nombre": to_name_from_node(k_best),
                "flujo_total_i_a_centro": fl_i
            })
    df_asigs = pd.DataFrame(asigs).sort_values(["centro_nombre","municipio_nombre"])
    df_asigs = _fix_names_in_df(df_asigs)

    # FLUJOS X
    rows_X = []
    for (i,k) in m.A:
        val = _v(m.X[(i,k), product])
        if abs(val) <= EPS: continue
        d = distOA[int(i)][int(k)]
        rows_X.append({
            "origen_nombre": to_name_from_node(i),
            "centro_nombre": to_name_from_node(k),
            "producto": product,
            "flujo": val,
            "dist_km": d
        })
    df_X = pd.DataFrame(rows_X).sort_values(["centro_nombre","origen_nombre"])
    df_X = _fix_names_in_df(df_X)

    # FLUJOS Y
    rows_Y = []
    for k in m.K:
        for j in m.J:
            val = _v(m.Y[k, j, product])
            if abs(val) <= EPS: continue
            d = distKD[int(k)][int(j)]
            rows_Y.append({
                "centro_nombre": to_name_from_node(k),
                "destino_nombre": to_name_from_node(j),
                "producto": product,
                "flujo": val,
                "dist_km": d
            })
    df_Y = pd.DataFrame(rows_Y).sort_values(["centro_nombre","destino_nombre"])
    df_Y = _fix_names_in_df(df_Y)

    obj_val = _v(m.OBJ) if hasattr(m, "OBJ") else None
    return df_centros, df_asigs, df_X, df_Y, obj_val

# ------------------------------------------------------------
# ESRI JSON -> GeoJSON reproyectado y mapa Folium (HTML)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
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
        props["MpCodigo_norm"] = _to_int_safe(mp)
        props["MpNombre"] = strip_accents(props.get("MpNombre") or props.get("MPNombre") or props.get("mpnombre") or "")
        geom = f.get("geometry") or {}
        if "rings" in geom:
            rings = geom["rings"]
            rings_ll = [reproj_ring(r) for r in rings]
            geom_gj = {"type": "Polygon", "coordinates": rings_ll}
        else:
            continue
        gj_features.append({"type": "Feature", "properties": props, "geometry": geom_gj})

    return {"type": "FeatureCollection", "features": gj_features}

@st.cache_data(show_spinner=False)
def build_map_html(geojson_fc, df_geo, df_asigs, df_centros, dane_to_name, dane_to_xy, name_to_dane):
    # Centro del mapa
    coords_all = [(float(r.lat), float(r.lon)) for _, r in df_geo.dropna(subset=["lat","lon"]).iterrows()]
    clat = sum(a for a,_ in coords_all)/len(coords_all) if coords_all else 2.93
    clon = sum(b for _,b in coords_all)/len(coords_all) if coords_all else -75.28
    fmap = folium.Map(location=[clat, clon], zoom_start=8, tiles="cartodbpositron")

    # Paleta
    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    centros_orden = sorted(df_centros["centro_nombre"].dropna().unique()) if not df_centros.empty else []
    color_by_centro = {c: palette[i % len(palette)] for i, c in enumerate(centros_orden)}

    # --- Join por DANE (seguro) ---
    # Mapa municipio_dane -> centro_dane (usando name_to_dane)
    cluster_by_dane = {}
    if not df_asigs.empty:
        for _, r in df_asigs.iterrows():
            mun_name = str(r["municipio_nombre"])
            cen_name = str(r["centro_nombre"])
            mun_dane = name_to_dane.get(mun_name)
            cen_dane = name_to_dane.get(cen_name)
            if mun_dane is not None and cen_dane is not None:
                cluster_by_dane[int(mun_dane)] = int(cen_dane)

    # Estilo por polígono usando MpCodigo_norm
    def style_fn(feat):
        dane = feat["properties"].get("MpCodigo_norm")
        if dane in cluster_by_dane:
            c_dane = cluster_by_dane[dane]
            c_name = dane_to_name.get(int(c_dane), "")
            c_name = strip_accents(c_name)
            return {"fillColor": color_by_centro.get(c_name, "#cccccc"), "color": "#555", "weight": 0.8, "fillOpacity": 0.6}
        else:
            return {"fillColor": "#ddd", "color": "#999", "weight": 0.5, "fillOpacity": 0.3}

    folium.GeoJson(
        geojson_fc,
        name="Clúster por polígono",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=["MpNombre","MpCodigo_norm"], aliases=["Municipio","DANE"])
    ).add_to(fmap)

    # Marcadores de centros (por DANE->XY)
    if not df_centros.empty:
        for _, r in df_centros.iterrows():
            c_name = str(r["centro_nombre"])
            c_dane = name_to_dane.get(c_name)
            if c_dane is None: continue
            lat, lon = dane_to_xy.get(int(c_dane), (None, None))
            if None in (lat,lon): continue
            folium.CircleMarker(
                location=[lat,lon], radius=7, color=color_by_centro.get(c_name, "#333"),
                fill=True, fill_opacity=0.95, popup=f"Centro: {c_name}"
            ).add_to(fmap)

    return fmap.get_root().render()

# ------------------------------------------------------------
# Curva costo vs p (1..15) usando appsi_highs
# ------------------------------------------------------------
def cost_curve_1_15(product, datos):
    points = []
    for p in range(1, 16):
        m, msgs, _ = build_and_solve(product, p, dmax=1e9, forced_danes=[], datos=datos, backend="appsi_highs")
        gc.collect(); time.sleep(0.05)
        if (m is None) or msgs:
            points.append({"p": p, "costo": np.nan, "status": "infeasible"})
            continue
        try:
            points.append({"p": p, "costo": float(_v(m.OBJ)), "status": "ok"})
        except Exception as e:
            points.append({"p": p, "costo": np.nan, "status": str(e)})
    return pd.DataFrame(points)

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("ZODAS – Optimización de Clústeres Agrícolas (Huila)")

datos = cached_load_data()
productos_validos = datos["productos_validos"] if datos["productos_validos"] else (datos["pact"] or datos["productos"])
if not productos_validos:
    st.error("No se encontraron productos válidos en Oferta∩Demanda. Revisa los .dat.")
    st.stop()

with st.sidebar:
    st.header("Parámetros de entrada")
    producto = st.selectbox("Producto", productos_validos, index=0)
    dmax = st.number_input("Distancia máxima O→A (km)", min_value=1.0, max_value=5000.0, value=150.0, step=1.0)
    p_max_allowed = min(P_MAX_UI, len(datos["K_ids"]))
    p = st.number_input("Cantidad de clústeres (p)", min_value=1, max_value=p_max_allowed,
                        value=min(datos["p_default"], p_max_allowed), step=1, help=f"Máximo permitido: {p_max_allowed}")

    # centroides forzados por NOMBRE (sin tildes)
    dane_to_name = datos["dane_to_name"]
    name_to_dane = datos["name_to_dane"]
    candidatos = sorted({strip_accents(dane_to_name.get(to_dane(k, datos["id_to_dane"], dane_to_name), f"ID {k}")) for k in datos["K_ids"]})

    forced_names = st.multiselect(f"Municipios que deben ser centros (0..{p})",
                                  options=candidatos, default=[],
                                  help="Opcional. Si p=15, selecciona máximo 15 municipios.")
    forced_danes = [name_to_dane[n] for n in forced_names if n in name_to_dane]
    if len(forced_danes) > p:
        st.error(f"Seleccionaste {len(forced_danes)} centros forzados pero p={p}. Reduce la cantidad o incrementa p.")

tabs = st.tabs(["Optimización", "Curva costo vs p"])

# ------------------------- TAB 1 -------------------------
with tabs[0]:
    prev_reasons = diagnostics(
        datos["oferta"], datos["demanda"], datos["cap"], datos["K_ids"],
        producto, int(p), float(dmax), datos["distOA"], datos["I_ids"], datos["J_ids"],
        datos["id_to_dane"], datos["dane_to_name"]
    )
    if prev_reasons:
        st.warning("**Chequeos previos** (posibles causas de infactibilidad):\n- " + "\n- ".join(prev_reasons))

    run_btn = st.button("Resolver modelo", type="primary")
    if run_btn:
        if len(forced_danes) > p:
            st.stop()
        with st.spinner("Resolviendo modelo..."):
            m, msgs, forced_k = build_and_solve(producto, int(p), float(dmax), forced_danes, datos, backend="highs")
            if (m is None) or msgs:
                m2, msgs2, _ = build_and_solve(producto, int(p), float(dmax), forced_danes, datos, backend="appsi_highs")
                if (m2 is not None) and not msgs2:
                    m, msgs = m2, []

        if (m is None) or msgs:
            st.error("Motivo por el cual es infactible. Por favor intente ampliando el número de clústeres (p) o la distancia máxima entre el origen y el acopio (DMAX).")
            for msg in msgs: st.info(msg)
            st.stop()

        df_centros, df_asigs, df_X, df_Y, obj_val = extract_results(m, datos, producto)

        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            st.metric("Costo total", f"{obj_val:,.2f}" if obj_val is not None else "—")
            st.metric("Centros abiertos", len(df_centros))
        with col2:
            st.metric("Arcos O→A con flujo", len(df_X))
        with col3:
            st.metric("Arcos A→D con flujo", len(df_Y))

        # --- Tablas (solo nombres) ---
        st.subheader("Centroides abiertos")
        cols_centros = ["centro_nombre","flujo_entrante_total","flujo_saliente_total","capacidad"]
        st.dataframe(df_centros[cols_centros], use_container_width=True)

        st.subheader("Asignaciones municipio → centro")
        cols_asigs = ["municipio_nombre","centro_nombre","flujo_total_i_a_centro"]
        st.dataframe(df_asigs[cols_asigs], use_container_width=True)

        with st.expander("Flujos O→A (X)"):
            cols_x = ["origen_nombre","centro_nombre","producto","flujo","dist_km"]
            st.dataframe(df_X[cols_x], use_container_width=True)

        with st.expander("Flujos A→D (Y)"):
            cols_y = ["centro_nombre","destino_nombre","producto","flujo","dist_km"]
            st.dataframe(df_Y[cols_y], use_container_width=True)

        # --- Descargas ---
        st.download_button("Descargar centroides (CSV)", df_centros[cols_centros].to_csv(index=False).encode("utf-8"), "result_centroides.csv", "text/csv")
        st.download_button("Descargar asignaciones (CSV)", df_asigs[cols_asigs].to_csv(index=False).encode("utf-8"), "result_asignaciones.csv", "text/csv")
        st.download_button("Descargar flujos O→A (CSV)", df_X[cols_x].to_csv(index=False).encode("utf-8"), "result_flujos_X.csv", "text/csv")
        st.download_button("Descargar flujos A→D (CSV)", df_Y[cols_y].to_csv(index=False).encode("utf-8"), "result_flujos_Y.csv", "text/csv")

        # --- Mapa ---
        st.subheader("Mapa de clústeres")
        esri_path = DATA_DIR / "huila_municipios.json"
        try:
            geo_fc = esri_to_geojson_reproject(esri_path)
            map_html = build_map_html(geo_fc, datos["df_geo"], df_asigs, df_centros,
                                      datos["dane_to_name"], datos["dane_to_xy"], datos["name_to_dane"])
            html(map_html, height=650)
        except Exception as e:
            st.error(f"No se pudo generar el mapa: {e}")

# ------------------------- TAB 2 -------------------------
with tabs[1]:
    st.write("Ejecuta el modelo desde **1 a 15 clústeres** (sin restricción de distancias) para un producto seleccionado.")
    producto_curve = st.selectbox("Producto para la curva", datos["productos_validos"], index=0, key="prod_curve")
    run_curve = st.button("Calcular curva costo vs p (1..15)", type="primary")

    if run_curve:
        with st.spinner("Calculando..."):
            df_curve = cost_curve_1_15(producto_curve, datos)
        st.dataframe(df_curve, use_container_width=True)
        if df_curve["costo"].notna().any():
            chart = alt.Chart(df_curve.dropna()).mark_line(point=True).encode(
                x=alt.X('p:Q', title='# de clústeres'),
                y=alt.Y('costo:Q', title='Costo total'),
                tooltip=['p','costo','status']
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No se obtuvieron costos válidos en el rango 1..15 (posible infactibilidad). Intenta con otro producto.")
