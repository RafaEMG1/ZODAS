import math
import pandas as pd
import numpy as np
import pulp

# -----------------------------
# Distancia Haversine (km)
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlbd = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlbd/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def build_distance_matrix(df):
    n = len(df)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            D[i, j] = haversine_km(df.loc[i,'lat'], df.loc[i,'lon'],
                                   df.loc[j,'lat'], df.loc[j,'lon'])
    return D

# -----------------------------
# Pre-validaciones (warnings)
# -----------------------------
def precheck_infeasibilities(df, D, R, S, p, fixed_ids):
    warnings = []
    n = len(df)

    if p < len(fixed_ids):
        warnings.append(f"p={p} es menor que el número de centroides fijos ({len(fixed_ids)}).")
        return True, warnings

    # Cobertura por R
    uncovered = []
    for i in range(n):
        if not any(D[i, j] <= R for j in range(n)):
            uncovered.append(df.loc[i, 'municipio'])
    if uncovered:
        warnings.append("Municipios sin candidato dentro de R: " + ", ".join(uncovered))
        return True, warnings

    # Separación S entre fijos
    for a in range(len(fixed_ids)):
        for b in range(a+1, len(fixed_ids)):
            j1, j2 = fixed_ids[a], fixed_ids[b]
            if D[j1, j2] < S:
                warnings.append(
                    f"Centros fijos '{df.loc[j1,'municipio']}' y '{df.loc[j2,'municipio']}' "
                    f"a {D[j1,j2]:.1f} km < S={S}."
                )
                return True, warnings

    # Cota máxima de centros por S (greedy empaquetado)
    remaining = set(range(n))
    chosen = []
    while remaining:
        j = remaining.pop()
        chosen.append(j)
        to_remove = {t for t in list(remaining) if D[j, t] < S}
        remaining -= to_remove
    max_centers_sep = len(chosen)
    if p > max_centers_sep:
        warnings.append(
            f"Con S={S} km, como máximo {max_centers_sep} centros respetan la separación mínima. p={p} es demasiado alto."
        )
        return True, warnings

    return False, warnings

# -----------------------------
# MIP p-mediana (PuLP/CBC)
# -----------------------------
def solve_p_median(df, p, R, S, fixed_centroids=None, time_limit=120, msg=False):
    df = df.reset_index(drop=True).copy()
    n = len(df)
    D = build_distance_matrix(df)
    w = df['produccion'].values.astype(float)

    # Mapear centroides fijos por nombre
    fixed_ids = []
    if fixed_centroids:
        name_to_id = {df.loc[j, 'municipio']: j for j in range(n)}
        for name in fixed_centroids:
            if name in name_to_id:
                fixed_ids.append(name_to_id[name])

    infeasible, warn = precheck_infeasibilities(df, D, R, S, p, fixed_ids)
    if infeasible:
        return None, None, None, warn + ["Problema infactible por pre-validación."]

    edges = [(i, j) for i in range(n) for j in range(n) if D[i, j] <= R]

    prob = pulp.LpProblem("p_median_zodas", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("y", (j for j in range(n)), lowBound=0, upBound=1, cat=pulp.LpBinary)
    x = pulp.LpVariable.dicts("x", (f"{i}_{j}" for (i, j) in edges), lowBound=0, upBound=1, cat=pulp.LpBinary)

    # Objetivo: ∑ w_i d_ij x_ij
    prob += pulp.lpSum(w[i] * D[i, j] * x[f"{i}_{j}"] for (i, j) in edges)

    # Asignación única
    for i in range(n):
        prob += pulp.lpSum(x[f"{i}_{j}"] for (ii, j) in edges if ii == i) == 1, f"assign_{i}"

    # Enlace x_ij <= y_j
    for (i, j) in edges:
        prob += x[f"{i}_{j}"] <= y[j]

    # Número de centros
    prob += pulp.lpSum(y[j] for j in range(n)) == p, "p_centers"

    # Separación mínima S entre centros
    for j1 in range(n):
        for j2 in range(j1+1, n):
            if D[j1, j2] < S:
                prob += y[j1] + y[j2] <= 1, f"sep_{j1}_{j2}"

    # Centroides fijos
    for j in fixed_ids:
        prob += y[j] == 1, f"fixed_{j}"

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    status = prob.solve(solver)
    status_str = pulp.LpStatus.get(prob.status, "Unknown")
    if status_str not in ["Optimal", "Not Solved", "Optimal Infeasible"]:
        return None, None, None, [f"Estado del solver: {status_str}. Ajuste p, R o S."]

    # Extraer solución
    open_centers = [j for j in range(n) if pulp.value(y[j]) > 0.5]
    assign = {i: None for i in range(n)}
    for (i, j) in edges:
        if pulp.value(x[f"{i}_{j}"]) > 0.5:
            assign[i] = j
    if any(v is None for v in assign.values()):
        return None, None, None, ["Asignaciones incompletas: revise R, S o aumente el tiempo."]

    cost = float(sum(w[i] * D[i, assign[i]] for i in range(n)))

    rows = []
    for i in range(n):
        j = assign[i]
        rows.append({
            "departamento": df.loc[i, "departamento"],
            "municipio": df.loc[i, "municipio"],
            "producto": df.loc[i, "producto"],
            "produccion": float(df.loc[i, "produccion"]),
            "lat": float(df.loc[i, "lat"]),
            "lon": float(df.loc[i, "lon"]),
            "centroide": df.loc[j, "municipio"],
            "centro_lat": float(df.loc[j, "lat"]),
            "centro_lon": float(df.loc[j, "lon"]),
        })
    df_assign = pd.DataFrame(rows)
    df_centros = (df_assign[["centroide","centro_lat","centro_lon"]]
                  .drop_duplicates().reset_index(drop=True))

    return df_assign, df_centros, cost, []

def cost_vs_p(df, Ps, R, S, fixed_centroids=None, time_limit=30, msg=False):
    out = []
    for p in Ps:
        dfA, dfC, cost, warns = solve_p_median(
            df, p, R, S, fixed_centroids, time_limit=time_limit, msg=msg
        )
        out.append({
            "p": p,
            "costo": float(cost) if cost is not None else float("nan"),
            "status": "ok" if not warns else "warning",
            "warning": "; ".join(warns) if warns else ""
        })
    return pd.DataFrame(out)
