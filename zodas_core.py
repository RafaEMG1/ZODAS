# zodas_core.py
from pathlib import Path
import re, csv, json
import pandas as pd
import numpy as np
import pyomo.environ as pyo

DATA_DIR = Path(__file__).parent / "data"

# ---------- Parsers AMPL .dat ----------
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

# ---------- CSV DANE/nombres/coords ----------
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
    df["municipio"] = df["municipio"].astype(str).str.strip()
    return df[["id_idx","dane_code","municipio","lat","lon"]].dropna(subset=["id_idx","dane_code"])

def build_maps_from_csv(df_geo: pd.DataFrame):
    id_to_dane   = {int(r.id_idx): int(r.dane_code) for _, r in df_geo.iterrows()}
    dane_to_name = {int(r.dane_code): str(r.municipio) for _, r in df_geo.iterrows()}
    dane_to_xy   = {int(r.dane_code): (float(r.lat), float(r.lon)) for _, r in df_geo.iterrows()}
    return id_to_dane, dane_to_name, dane_to_xy

def to_dane(x, id_to_dane, dane_to_name):
    x = int(x)
    if x in id_to_dane: return id_to_dane[x]
    if x in dane_to_name: return x
    return x

# ---------- Carga de datos Huila ----------
def load_huila_data():
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
    assert set(J_ids) == set(colsKD),  "Columnas DistanciaAcopioDestino deben coincidir con Demanda (IDs)"

    df_geo = load_huila_csv(csv_p)
    id_to_dane, dane_to_name, dane_to_xy = build_maps_from_csv(df_geo)

    return dict(
        productos=productos, pact=pact, cap=cap, p_default=p_default,
        ctr=ctr, ca=ca, tonco2=tonco2,
        oferta=oferta, demanda=demanda, distOA=distOA, distKD=distKD,
        I_ids=I_ids, J_ids=J_ids, K_ids=K_ids,
        df_geo=df_geo, id_to_dane=id_to_dane, dane_to_name=dane_to_name, dane_to_xy=dane_to_xy
    )

# ---------- Diagnósticos previos ----------
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
        names = [f"{dane_to_name.get(to_dane(i,id_to_dane,dane_to_name),'?')} (DANE {to_dane(i,id_to_dane,dane_to_name)})" for i in unreachable_i[:10]]
        extra = " ..." if len(unreachable_i) > 10 else ""
        reasons.append(f"{len(unreachable_i)} municipio(s) sin acopio dentro de DMAX={dmax} km: " + ", ".join(names) + extra)

    return reasons

# ---------- Modelo Pyomo ----------
def build_and_solve(product, p, dmax, forced_danes, datos):
    oferta  = datos["oferta"]; demanda = datos["demanda"]
    distOA  = datos["distOA"]; distKD = datos["distKD"]
    cap     = datos["cap"]
    I_ids   = datos["I_ids"]; J_ids = datos["J_ids"]; K_ids = datos["K_ids"]
    ctr     = datos["ctr"]; ca = datos["ca"]; tonco2 = datos["tonco2"]
    id_to_dane = datos["id_to_dane"]; dane_to_name = datos["dane_to_name"]

    # arcos permitidos O→A por DMAX
    A_allowed = {(i,k) for i in I_ids for k in K_ids if distOA[i][k] <= float(dmax) + 1e-9}

    # mapear centros forzados (DANE) a k internos
    def dane_to_k(dane):
        for k in K_ids:
            if to_dane(k, id_to_dane, dane_to_name) == int(dane):
                return k
        return None

    forced_k = [dane_to_k(int(d)) for d in forced_danes]
    forced_k = [k for k in forced_k if k is not None]

    # checks rápidos
    if len(forced_k) > p:
        return None, ["El número de centroides forzados excede p."], None

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
        return sum(_m.X[(i,k), product] for (i,kk) in _m.A if kk==k) <= cap[k]*_m.T[k]
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

    # forzar centros
    for k in forced_k:
        m.add_component(f"Force_{k}", pyo.Constraint(expr=m.T[k] == 1))

    # solve (HiGHS)
    status, msg = None, []
    try:
        solver = pyo.SolverFactory("highs")
        solver.options["time_limit"] = 180
        solver.options["mip_rel_gap"] = 0.01
        solver.options["presolve"] = "on"
    except Exception:
        # fallback appsi
        solver = pyo.SolverFactory("appsi_highs")

    res = solver.solve(m, tee=False)
    status = str(res.solver.termination_condition)

    if status.lower() not in ("optimal", "optimal termination", "feasible"):
        msg.append("No se encontró solución óptima factible.")

    return m, msg, forced_k

# ---------- Extracción de resultados ----------
def extract_results(m, datos, product):
    id_to_dane = datos["id_to_dane"]; dane_to_name = datos["dane_to_name"]
    distOA = datos["distOA"]; distKD = datos["distKD"]; cap = datos["cap"]
    EPS = 1e-6

    def to_name(d):
        return dane_to_name.get(d, f"ID {d}")

    # centros
    centros = []
    for k in m.K:
        if pyo.value(m.T[k]) > 0.5:
            fin = sum(pyo.value(m.X[(i,k), product]) for i in m.I if (i,k) in m.A)
            fout= sum(pyo.value(m.Y[k, j, product]) for j in m.J)
            top_i, best = None, -1.0
            for i in m.I:
                if (i,k) not in m.A: continue
                val = pyo.value(m.X[(i,k), product])
                if val > best: best, top_i = val, i
            k_dane = to_dane(k, id_to_dane, dane_to_name)
            top_dane = to_dane(top_i, id_to_dane, dane_to_name) if top_i is not None else None
            centros.append({
                "centro_dane": k_dane,
                "centro_nombre": to_name(k_dane),
                "municipio_centro_dane": top_dane,
                "municipio_centro_nombre": to_name(top_dane) if top_dane else None,
                "flujo_entrante_total": fin,
                "flujo_saliente_total": fout,
                "capacidad": cap.get(k, np.nan),
            })
    df_centros = pd.DataFrame(centros).sort_values("centro_dane")

    # asignaciones
    asigs = []
    for i in m.I:
        ks = [k for k in m.K if (i,k) in m.A and pyo.value(m.U[(i,k)]) > 0.5]
        if ks:
            k_best = max(ks, key=lambda kk: pyo.value(m.X[(i,kk), product]))
            fl_i = pyo.value(m.X[(i,k_best), product])
            i_dane = to_dane(i, id_to_dane, dane_to_name)
            k_dane = to_dane(k_best, id_to_dane, dane_to_name)
            asigs.append({
                "municipio_dane": i_dane, "municipio_nombre": to_name(i_dane),
                "centro_dane": k_dane, "centro_nombre": to_name(k_dane),
                "flujo_total_i_a_centro": fl_i
            })
    df_asigs = pd.DataFrame(asigs).sort_values(["centro_dane","municipio_dane"])

    # flujos X
    rows_X = []
    for (i,k) in m.A:
        val = pyo.value(m.X[(i,k), product])
        if abs(val) <= EPS: continue
        d = distOA[i][k]
        cu = 0.0  # costo unitario ya no es imprescindible aquí
        i_d = to_dane(i, id_to_dane, dane_to_name)
        k_d = to_dane(k, id_to_dane, dane_to_name)
        rows_X.append({
            "origen_dane": i_d, "origen_nombre": to_name(i_d),
            "centro_dane": k_d, "centro_nombre": to_name(k_d),
            "producto": product, "flujo": val, "dist_km": d
        })
    df_X = pd.DataFrame(rows_X).sort_values(["centro_dane","origen_dane"])

    # flujos Y
    rows_Y = []
    for k in m.K:
        for j in m.J:
            val = pyo.value(m.Y[k, j, product])
            if abs(val) <= EPS: continue
            d = distKD[k][j]
            j_d = to_dane(j, id_to_dane, dane_to_name)
            k_d = to_dane(k, id_to_dane, dane_to_name)
            rows_Y.append({
                "centro_dane": k_d, "centro_nombre": to_name(k_d),
                "destino_dane": j_d, "destino_nombre": to_name(j_d),
                "producto": product, "flujo": val, "dist_km": d
            })
    df_Y = pd.DataFrame(rows_Y).sort_values(["centro_dane","destino_dane"])

    obj_val = pyo.value(m.OBJ) if hasattr(m, "OBJ") else None
    return df_centros, df_asigs, df_X, df_Y, obj_val

# ---------- Curva costo vs p ----------
def cost_curve(p_list, product, dmax, forced_danes, datos):
    points = []
    for p in p_list:
        m, msgs, _ = build_and_solve(product, p, dmax, forced_danes, datos)
        if m is None:
            points.append({"p": p, "costo": np.nan, "status": "; ".join(msgs)})
            continue
        try:
            costo = pyo.value(m.OBJ)
            points.append({"p": p, "costo": costo, "status": "ok"})
        except Exception as e:
            points.append({"p": p, "costo": np.nan, "status": str(e)})
    return pd.DataFrame(points)
