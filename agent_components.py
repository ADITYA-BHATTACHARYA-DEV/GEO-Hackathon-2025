# agent_components.py

import math
import numpy as np
import pandas as pd
import re

# ---------- Defaults ----------
DEFAULTS = {
    "rho": 1000.0,
    "mu": 1e-3,
    "roughness": 1e-5,
    "reservoir_pressure": 230.0,
    "wellhead_pressure": 10.0,
    "PI": 5.0,
    "esp_depth": 500.0,
    "pump_curve": {"flow": [0, 100, 200, 300, 400], "head": [600, 550, 450, 300, 100]},
    "trajectory": [
        {"MD": 0.0, "TVD": 0.0, "ID": 0.3397},
        {"MD": 500.0, "TVD": 500.0, "ID": 0.2445},
        {"MD": 1500.0, "TVD": 1500.0, "ID": 0.1778},
        {"MD": 2500.0, "TVD": 2500.0, "ID": 0.1778},
    ],
}

# ---------- Parsers ----------
def parse_scalar_from_text(text, key):
    if not text:
        return None
    t = text.lower()
    patterns = {
        "reservoir_pressure": [r"reservoir\s+pressure[^0-9]*([\d\.]+)"],
        "wellhead_pressure":  [r"wellhead\s+pressure[^0-9]*([\d\.]+)"],
        "PI":                 [r"(productivity\s+index|pi)[^0-9]*([\d\.]+)"],
        "esp_depth":          [r"(esp\s+(?:intake\s+)?depth)[^0-9]*([\d\.]+)"],
        "rho":                [r"(density|rho)[^0-9]*([\d\.]+)"],
        "mu":                 [r"(viscosity|mu)[^0-9]*([\d\.eE\-]+)"],
        "roughness":          [r"(roughness)[^0-9]*([\d\.eE\-]+)"],
    }.get(key, [])
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            num = m.groups()[-1]
            try:
                return float(num)
            except:
                continue
    return None

def parse_pump_curve_from_csv(csv_text):
    try:
        df = pd.read_csv(pd.io.common.StringIO(csv_text))
        flow_col = next((c for c in df.columns if "flow" in c.lower()), None)
        head_col = next((c for c in df.columns if "head" in c.lower()), None)
        if flow_col and head_col:
            flows = df[flow_col].astype(float).tolist()
            heads = df[head_col].astype(float).tolist()
            return {"flow": flows, "head": heads}
    except Exception:
        pass
    return None

def parse_trajectory_from_csv(csv_text):
    try:
        df = pd.read_csv(pd.io.common.StringIO(csv_text))
        md_col = next((c for c in df.columns if c.strip().upper() == "MD"), None)
        tvd_col = next((c for c in df.columns if c.strip().upper() == "TVD"), None)
        id_col = next((c for c in df.columns if c.strip().upper() in {"ID", "DIAMETER", "INNER DIAMETER"}), None)
        if md_col and tvd_col and id_col:
            rows = []
            for _, r in df.iterrows():
                try:
                    rows.append({"MD": float(r[md_col]), "TVD": float(r[tvd_col]), "ID": float(r[id_col])})
                except Exception:
                    continue
            if rows:
                return sorted(rows, key=lambda x: x["MD"])
    except Exception:
        pass
    return None

# ---------- RAG aggregation ----------
def get_parameters_from_rag(source_docs):
    params = {k: None for k in DEFAULTS.keys()}
    for doc in source_docs or []:
        md = getattr(doc, "metadata", {}) or {}
        text = getattr(doc, "page_content", "") or ""

        # direct metadata
        for k in ["rho", "mu", "roughness", "reservoir_pressure", "wellhead_pressure", "PI", "esp_depth"]:
            if params[k] is None and md.get(k) is not None:
                try:
                    params[k] = float(md[k])
                except Exception:
                    pass

        # pump curve from metadata or table
        if params["pump_curve"] is None:
            if md.get("pump_curve"):
                pc = md["pump_curve"]
                if isinstance(pc, dict) and "flow" in pc and "head" in pc:
                    try:
                        params["pump_curve"] = {
                            "flow": list(map(float, pc["flow"])),
                            "head": list(map(float, pc["head"]))
                        }
                    except Exception:
                        pass
            elif md.get("table_csv"):
                pc = parse_pump_curve_from_csv(md["table_csv"])
                if pc:
                    params["pump_curve"] = pc

        # trajectory from metadata or table
        if params["trajectory"] is None:
            if md.get("trajectory"):
                traj = md["trajectory"]
                if isinstance(traj, list) and traj and all(set(["MD","TVD","ID"]).issubset(t.keys()) for t in traj):
                    try:
                        params["trajectory"] = [
                            {"MD": float(t["MD"]), "TVD": float(t["TVD"]), "ID": float(t["ID"])} for t in traj
                        ]
                    except Exception:
                        pass
            elif md.get("table_csv"):
                traj = parse_trajectory_from_csv(md["table_csv"])
                if traj:
                    params["trajectory"] = traj

        # heuristics from text
        for k in ["reservoir_pressure", "wellhead_pressure", "PI", "esp_depth", "rho", "mu", "roughness"]:
            if params[k] is None:
                val = parse_scalar_from_text(text, k)
                if val is not None:
                    params[k] = val

    # defaults for missing values
    for k, v in DEFAULTS.items():
        if params.get(k) is None:
            params[k] = v

    # basic validation clamps
    params["rho"] = max(params["rho"], 1.0)
    params["mu"] = max(params["mu"], 1e-6)
    params["roughness"] = max(params["roughness"], 0.0)
    params["PI"] = max(params["PI"], 1e-6)
    if not params["pump_curve"] or len(params["pump_curve"]["flow"]) < 2:
        params["pump_curve"] = DEFAULTS["pump_curve"]
    if not params["trajectory"] or len(params["trajectory"]) < 2:
        params["trajectory"] = DEFAULTS["trajectory"]

    return params

# ---------- Nodal analysis ----------
def build_segments(trajectory):
    segments = []
    for i in range(1, len(trajectory)):
        MD = trajectory[i]["MD"] - trajectory[i-1]["MD"]
        TVD = trajectory[i]["TVD"] - trajectory[i-1]["TVD"]
        D = trajectory[i]["ID"]
        L = MD
        theta = math.atan2(TVD, MD if MD != 0 else 1e-9)
        segments.append((L, D, theta))
    return segments

def swamee_jain(Re, D, roughness):
    if Re <= 0:
        return 0.0
    return 0.25 / (math.log10((roughness/(3.7*D)) + (5.74/(Re**0.9))))**2

def pump_interp(flow, pump_curve, key):
    return np.interp(flow, pump_curve["flow"], pump_curve[key])

def vlp(flow_m3hr, segments, rho, mu, g, roughness, esp_depth, pump_curve, wellhead_pressure):
    q = flow_m3hr / 3600.0
    dp_total = 0.0
    depth_accum = 0.0
    for (L, D, theta) in segments:
        A = math.pi * D**2 / 4.0
        u = q / A if A > 0 else 0.0
        Re = rho * abs(u) * D / mu if mu > 0 else 0.0
        f = swamee_jain(Re, D, roughness)

        dp_fric = f * (L / max(D, 1e-9)) * (rho * u**2 / 2.0)
        dp_grav = rho * 9.81 * L * math.sin(theta)
        dp_total += dp_fric + dp_grav
        depth_accum += L * math.sin(theta)

    if depth_accum >= esp_depth and pump_curve:
        dp_total -= rho * 9.81 * pump_interp(flow_m3hr, pump_curve, "head")

    return wellhead_pressure + dp_total / 1e5

def ipr(flow_m3hr, reservoir_pressure, PI):
    pbh = reservoir_pressure - flow_m3hr / max(PI, 1e-9)
    return max(pbh, 0.0)

def run_nodal_analysis(params, flow_min=1.0, flow_max=400.0, n_points=200, tolerance_bar=3.0):
    segments = build_segments(params["trajectory"])
    flows = np.linspace(flow_min, flow_max, n_points)
    p_vlp = np.array([
        vlp(f, segments, params["rho"], params["mu"], 9.81, params["roughness"],
            params["esp_depth"], params["pump_curve"], params["wellhead_pressure"])
        for f in flows
    ])
    p_ipr = np.array([ipr(f, params["reservoir_pressure"], params["PI"]) for f in flows])

    diff = np.abs(p_vlp - p_ipr)
    idx = np.argmin(diff)
    if diff[idx] < tolerance_bar:
        sol_flow = flows[idx]
        sol_pbh = p_vlp[idx]
        sol_head = pump_interp(sol_flow, params["pump_curve"], "head") if params["pump_curve"] else None
    else:
        sol_flow = sol_pbh = sol_head = None

    return {
        "sol_flow": sol_flow,
        "sol_pbh": sol_pbh,
        "sol_head": sol_head,
        "flows": flows,
        "p_vlp": p_vlp,
        "p_ipr": p_ipr
    }

