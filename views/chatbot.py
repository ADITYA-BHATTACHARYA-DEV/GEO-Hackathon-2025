# import selected
# import selected
import plt
import streamlit as st
from streamlit_option_menu import option_menu
import os
from pathlib import Path
import re
import math
import numpy as np
from io import StringIO

import streamlit as st
from pathlib import Path
import requests
import base64
import json
from fpdf import FPDF
from PIL import Image
# Core prompt template
from langchain_core.prompts import PromptTemplate

# Memory + Chains (LangChain 1.x with classic imports)
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import RetrievalQA

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Community integrations
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama

# Document loaders with defensive imports
from langchain_community.document_loaders import PyPDFLoader

st.header("LocoChat - Document insights across text, tables, and figures")

def safe_float_value(extracted_data, key, default_value):
    """Safely converts an extracted parameter to float, using a default on failure."""
    try:
        # Prioritize extracted data if it's not None
        value = extracted_data.get(key)
        if value is not None:
            return float(value)
        # Fallback to provided default value if extracted is None
        return float(default_value)
    except (TypeError, ValueError):
        # Catch conversion errors (e.g., trying to float("NaN") or float("text"))
        return float(default_value)
import re

import re
from datetime import datetime
from pathlib import Path
from fpdf import FPDF

# def safe_text(text, max_len=400):
#     """Break long words and truncate text for safe PDF rendering."""
#     if not text:
#         return ""
#     text = str(text).replace("\n", " ")
#     # Insert spaces into very long words (every 80 chars)
#     safe = re.sub(r"(\S{80})(?=\S)", r"\1 ", text)
#     return safe[:max_len] + ("..." if len(text) > max_len else "")
#
# def safe_multicell(pdf, w, h, text, max_len=400):
#     """Wrapper around multi_cell to prevent FPDFException crashes."""
#     try:
#         pdf.multi_cell(w, h, safe_text(text, max_len=max_len))
#     except Exception:
#         # Fallback: render clipped text in a single cell
#         pdf.cell(w, h, safe_text(text, max_len=100), ln=True)




import re
from datetime import datetime
from pathlib import Path
from fpdf import FPDF

def safe_text(text, max_len=400):
    """Break long words and truncate text for safe PDF rendering."""
    if not text:
        return ""
    text = str(text).replace("\n", " ")
    safe = re.sub(r"(\S{80})(?=\S)", r"\1 ", text)
    return safe[:max_len] + ("..." if len(text) > max_len else "")

def safe_multicell(pdf, w, h, text, max_len=400):
    """Wrapper around multi_cell to prevent FPDFException crashes."""
    try:
        pdf.multi_cell(w, h, safe_text(text, max_len=max_len))
    except Exception:
        pdf.cell(w, h, safe_text(text, max_len=100), ln=True)


import numpy as np
from fpdf import FPDF
from io import StringIO
from datetime import datetime
from pathlib import Path


# Assume safe_text and safe_multicell are defined globally and handle text wrapping.
from fpdf import FPDF
from datetime import datetime
from io import StringIO
import pandas as pd
import os

def save_response_to_pdf(
    answer_text,
    source_docs,
    filename="output.pdf",
    user_query=None,
    operating_point=None,      # dict
    parameter_dict=None,       # dict
    nodal_plot_path=None
):
    """
    SPE-style PDF generator:
      - Vertical SPE-style tables (Option A)
      - Dark-gray header, light-gray body
      - Nodal plot insertion
      - Full Unicode-safe text handling
      - Fixes FPDF crash: "'int' object has no attribute 'upper'"
    """

    # -----------------------------
    # Imports
    # -----------------------------
    from fpdf import FPDF
    import os
    from datetime import datetime

    # -----------------------------
    # Safe text conversion (CRITICAL)
    # -----------------------------
    def s(x):
        """Convert any object to safe string for FPDF."""
        try:
            if x is None:
                return ""
            return str(x)
        except:
            return ""

    # -----------------------------
    # Safe multicell wrapper
    # -----------------------------
    def _safe_multicell(pdf_obj, w, h, txt, max_len=3000):
        txt = s(txt)
        if len(txt) > max_len:
            txt = txt[:max_len] + "..."
        pdf_obj.multi_cell(w, h, txt)

    # -----------------------------
    # Custom PDF class
    # -----------------------------
    class _PEFPDF(FPDF):
        def footer(self):
            self.set_y(-12)
            self.set_font("Arial", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = _PEFPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # -----------------------------
    # SPE Colors
    # -----------------------------
    HEADER_GRAY = (80, 80, 80)      # dark gray header
    BODY_GRAY = (235, 235, 235)     # light gray body
    TEXT_BLACK = (0, 0, 0)

    # -----------------------------
    # Heading formatter
    # -----------------------------
    def draw_heading(title):
        pdf.ln(6)
        pdf.set_font("Arial", "B", 14)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(0, 8, s(title), ln=True, align="L")
        pdf.ln(2)
        pdf.set_draw_color(150, 150, 150)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.l_margin, pdf.get_y())
        pdf.ln(4)

    # -----------------------------
    # SPE-style vertical table
    # -----------------------------
    def draw_spe_vertical_table(title, data_dict):
        if not data_dict:
            return

        pdf.ln(3)

        # ----------------- Header row -----------------
        pdf.set_fill_color(*HEADER_GRAY)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 8, s(title), ln=True, align="C", fill=True)

        # ----------------- Body rows -----------------
        pdf.set_fill_color(*BODY_GRAY)
        pdf.set_text_color(*TEXT_BLACK)
        pdf.set_font("Arial", "", 9)

        col1_w = (pdf.w - pdf.l_margin - pdf.r_margin) * 0.40
        col2_w = (pdf.w - pdf.l_margin - pdf.r_margin) * 0.60

        for key, val in data_dict.items():
            pdf.cell(col1_w, 7, s(key), border=1, fill=True)
            pdf.cell(col2_w, 7, s(val), border=1, fill=True, ln=1)

        pdf.ln(4)

    # -----------------------------
    # COVER PAGE
    # -----------------------------
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(10, 60, 150)
    pdf.cell(0, 12, "Nodal Analysis - Engineering Report", ln=True)

    pdf.ln(2)
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(*TEXT_BLACK)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 6, s(f"Generated: {ts}"), ln=True)

    if user_query:
        _safe_multicell(pdf, 0, 6, f"Query: {user_query}")

    pdf.ln(4)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.l_margin, pdf.get_y())
    pdf.ln(8)

    # -----------------------------
    # I. Summary
    # -----------------------------
    draw_heading("I. Summary of Solution")
    pdf.set_font("Arial", "", 11)
    _safe_multicell(pdf, 0, 5, answer_text)
    pdf.ln(6)

    # -----------------------------
    # II. Operating Point (SPE table)
    # -----------------------------
    if operating_point:
        draw_heading("II. Operating Point")
        draw_spe_vertical_table("Operating Point", operating_point)

    # -----------------------------
    # III. Extracted Parameters (SPE table)
    # -----------------------------
    if parameter_dict:
        draw_heading("III. Extracted Parameters")
        draw_spe_vertical_table("Extracted Parameters", parameter_dict)

    # -----------------------------
    # IV. Insert Nodal Plot
    # -----------------------------
    if nodal_plot_path:
        draw_heading("IV. Nodal Plot")
        try:
            if os.path.exists(nodal_plot_path):
                max_w = pdf.w - pdf.l_margin - pdf.r_margin
                pdf.image(nodal_plot_path, x=pdf.l_margin, w=max_w)
                pdf.ln(6)
            else:
                pdf.set_text_color(180, 0, 0)
                pdf.cell(0, 6, "Plot image not found.", ln=True)
                pdf.set_text_color(*TEXT_BLACK)
        except Exception as e:
            pdf.set_text_color(180, 0, 0)
            _safe_multicell(pdf, 0, 5, f"[Could not load plot: {e}]")
            pdf.set_text_color(*TEXT_BLACK)

    # -----------------------------
    # V. Supporting Sources
    # -----------------------------
    draw_heading("V. Supporting Sources")

    for i, doc in enumerate(source_docs or [], start=1):
        meta = getattr(doc, "metadata", {}) or {}

        pdf.set_fill_color(220, 220, 220)
        pdf.set_font("Arial", "B", 9)
        pdf.set_text_color(*TEXT_BLACK)

        header_text = f"Source {i}: {meta.get('source','N/A')} | Page: {meta.get('page','N/A')}"
        pdf.cell(0, 7, s(header_text), ln=1, fill=True)

        pdf.set_font("Arial", "", 8)
        preview = getattr(doc, "page_content", "")
        _safe_multicell(pdf, 0, 4, preview)
        pdf.ln(2)

    # -----------------------------
    # Save Output
    # -----------------------------
    out_dir = os.path.dirname(filename)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    pdf.output(filename)

    return filename




# def save_response_to_pdf(answer_text, source_docs, filename="output.pdf", user_query=None):
#     pdf = FPDF()
#     pdf.add_page()
#
#     # Define Colors (RGB)
#     COLOR_GRAY = (240, 240, 240)
#     COLOR_YELLOW = (255, 255, 220)
#     COLOR_BLACK = (0, 0, 0)
#
#     current_color = COLOR_YELLOW
#
#     # Helper function to render a block with a background color
#     def render_colored_block(text, font_size, is_heading=False):
#         nonlocal current_color
#
#         # Switch color for the next block
#         current_color = COLOR_GRAY if current_color == COLOR_YELLOW else COLOR_YELLOW
#         pdf.set_fill_color(*current_color)
#
#         # Calculate height required for the text block
#         # Temporarily set font to calculate height (max width is 0, so it uses page width minus margins)
#         pdf.set_font("Arial", 'B' if is_heading else '', font_size)
#         text_height = pdf.get_string_height(0, text)
#
#         # Add padding (e.g., 2 units)
#         block_height = text_height + 4
#
#         # Get current x and y positions
#         x = pdf.get_x()
#         y = pdf.get_y()
#
#         # Draw the background rectangle
#         pdf.rect(pdf.l_margin, y, pdf.w - 2 * pdf.l_margin, block_height, 'F')
#
#         # Set text color to black
#         pdf.set_text_color(*COLOR_BLACK)
#
#         # Render text inside the colored block
#         pdf.set_y(y + 2)  # Start text slightly below the top of the block
#         pdf.set_x(x)
#         safe_multicell(pdf, 0, block_height - 4, text, max_len=2000)
#
#         # Move cursor below the block
#         pdf.set_y(y + block_height)
#         pdf.ln(2)
#
#     # --- Header (No background) ---
#     pdf.set_text_color(*COLOR_BLACK)
#     pdf.set_font("Arial", size=10)
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     header = f"Report Generated: {timestamp}\n"
#     if user_query:
#         header += f"Query Context: {user_query}\n"
#     safe_multicell(pdf, 0, 5, header, max_len=500)
#     pdf.ln(5)
#
#     # --- Main Answer/Summary Section ---
#     render_colored_block(answer_text, 10)
#
#     # --- Sources Section ---
#     pdf.set_font("Arial", 'B', 12)
#     pdf.cell(0, 10, "III. Supporting Sources:", ln=True)
#
#     for i, doc in enumerate(source_docs or [], start=1):
#         md = getattr(doc, "metadata", {}) or {}
#
#         # Build Source Metadata String
#         meta_text = f"Source {i}: {md.get('source', 'unknown')} | Type: {md.get('content_type', 'N/A')} | Page: {md.get('page', 'N/A')}"
#
#         # Render Source Metadata (Colored Block)
#         render_colored_block(meta_text, 8, is_heading=True)
#
#         pdf.set_text_color(*COLOR_BLACK)
#         pdf.set_font("Arial", size=8)
#
#         # Render Table Preview or Raw Content (Inside current block color)
#         pdf.set_fill_color(*current_color)
#         pdf.set_font("Arial", size=8)
#
#         # Embed table if available (Using FPDF's internal cell logic for tables for control)
#         if md.get("table_csv"):
#             try:
#                 import pandas as pd
#                 df = pd.read_csv(StringIO(md["table_csv"]))
#
#                 pdf.set_font("Arial", 'B', 8)
#                 pdf.cell(0, 4, "Table Preview (First 3 Rows):", ln=True, fill=True)
#                 pdf.set_font("Arial", size=6)
#
#                 # Render Headers (Max 5 columns for layout safety)
#                 header_str = " | ".join(df.columns[:5].astype(str).tolist())
#                 safe_multicell(pdf, 0, 3, header_str, max_len=300)
#
#                 # Render Rows (Max 3 rows, Max 5 columns)
#                 for _, row in df.head(3).iterrows():
#                     row_str = " | ".join(map(lambda x: str(x)[:20], row.values[:5]))  # Truncate cell content
#                     safe_multicell(pdf, 0, 3, row_str, max_len=300)
#                 pdf.ln(1)
#             except Exception:
#                 pdf.set_font("Arial", size=8)
#                 preview = getattr(doc, "page_content", "")
#                 safe_multicell(pdf, 0, 4, "Raw Content Preview: " + preview, max_len=400)
#         else:
#             # Render standard text content preview
#             preview = getattr(doc, "page_content", "")
#             safe_multicell(pdf, 0, 4, preview, max_len=400)
#
#         pdf.ln(2)  # Add space after source block
#
#     pdf.output(filename)
#     return filename


####

import math
import numpy as np
import pandas as pd
import re

# Defaults used when RAG doesn't provide a value
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

def parse_scalar_from_text(text, key, unit_hint=None):
    """
    Heuristically extract scalar like 'Reservoir pressure = 230 bar'.
    key examples: 'reservoir pressure', 'PI', 'wellhead pressure', 'ESP depth', 'density', 'viscosity', 'roughness'
    """
    if not text:
        return None
    t = text.lower()
    patterns = []
    if key == "reservoir_pressure":
        patterns = [r"reservoir\s+pressure[^0-9]*([\d\.]+)"]
    elif key == "wellhead_pressure":
        patterns = [r"wellhead\s+pressure[^0-9]*([\d\.]+)"]
    elif key == "PI":
        patterns = [r"(productivity\s+index|pi)[^0-9]*([\d\.]+)"]
    elif key == "esp_depth":
        patterns = [r"(esp\s+(?:intake\s+)?depth)[^0-9]*([\d\.]+)"]
    elif key == "rho":
        patterns = [r"(density|rho)[^0-9]*([\d\.]+)"]
    elif key == "mu":
        patterns = [r"(viscosity|mu)[^0-9]*([\d\.]+)"]
    elif key == "roughness":
        patterns = [r"(roughness)[^0-9]*([\d\.eE\-]+)"]
    else:
        return None

    for pat in patterns:
        m = re.search(pat, t)
        if m:
            # last capturing group is the number
            num = m.groups()[-1]
            try:
                return float(num)
            except:
                continue
    return None

def parse_pump_curve_from_csv(csv_text):
    """Parse a pump curve from a CSV string with columns like Flow, Head."""
    try:
        df = pd.read_csv(pd.io.common.StringIO(csv_text))
        # Find columns by fuzzy name
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
    """Parse trajectory rows with columns MD, TVD, ID."""
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
                # Ensure sorted by MD
                rows = sorted(rows, key=lambda x: x["MD"])
                return rows
    except Exception:
        pass
    return None

def get_parameters_from_rag(source_docs):
    """
    Aggregate parameters from doc metadata and text. Fallback to DEFAULTS if not found.
    source_docs: list of Document objects with .metadata and .page_content
    """
    params = {
        "rho": None, "mu": None, "roughness": None,
        "reservoir_pressure": None, "wellhead_pressure": None, "PI": None, "esp_depth": None,
        "pump_curve": None, "trajectory": None,
    }

    for doc in source_docs or []:
        md = getattr(doc, "metadata", {}) or {}
        text = getattr(doc, "page_content", "") or ""

        # Prefer explicit metadata if present
        for k in ["rho", "mu", "roughness", "reservoir_pressure", "wellhead_pressure", "PI", "esp_depth"]:
            if params[k] is None and md.get(k) is not None:
                try:
                    params[k] = float(md[k])
                except Exception:
                    pass

        if params["pump_curve"] is None:
            if md.get("pump_curve"):
                pc = md["pump_curve"]
                if isinstance(pc, dict) and "flow" in pc and "head" in pc:
                    params["pump_curve"] = {"flow": list(map(float, pc["flow"])), "head": list(map(float, pc["head"]))}
            elif md.get("table_csv"):
                pc = parse_pump_curve_from_csv(md["table_csv"])
                if pc:
                    params["pump_curve"] = pc

        if params["trajectory"] is None:
            if md.get("trajectory"):
                traj = md["trajectory"]
                if isinstance(traj, list) and traj and all(set(["MD","TVD","ID"]).issubset(t.keys()) for t in traj):
                    # Normalize types
                    params["trajectory"] = [
                        {"MD": float(t["MD"]), "TVD": float(t["TVD"]), "ID": float(t["ID"])} for t in traj
                    ]
            elif md.get("table_csv"):
                traj = parse_trajectory_from_csv(md["table_csv"])
                if traj:
                    params["trajectory"] = traj

        # Heuristic extraction from text
        for k in ["reservoir_pressure", "wellhead_pressure", "PI", "esp_depth", "rho", "mu", "roughness"]:
            if params[k] is None:
                val = parse_scalar_from_text(text, k)
                if val is not None:
                    params[k] = val

    # Apply defaults for anything still missing
    for k, v in DEFAULTS.items():
        if params.get(k) is None:
            params[k] = v

    return params


def build_segments(trajectory):
    """Build (L, D, theta) segments from MD/TVD/ID points."""
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
    q = flow_m3hr / 3600.0  # m3/hr to m3/s
    dp_total = 0.0
    depth_accum = 0.0
    for (L, D, theta) in segments:
        A = math.pi * D**2 / 4.0
        u = q / A if A > 0 else 0.0
        Re = rho * abs(u) * D / mu if mu > 0 else 0.0
        f = swamee_jain(Re, D, roughness)

        dp_fric = f * (L / max(D, 1e-9)) * (rho * u**2 / 2.0)
        dp_grav = rho * g * L * math.sin(theta)
        dp_total += dp_fric + dp_grav
        depth_accum += L * math.sin(theta)

    # Pump head reduces required pressure if intake is below total vertical depth
    if depth_accum >= esp_depth and pump_curve:
        dp_total -= rho * g * pump_interp(flow_m3hr, pump_curve, "head")

    return wellhead_pressure + dp_total / 1e5  # Pa to bar

def ipr(flow_m3hr, reservoir_pressure, PI):
    pbh = reservoir_pressure - flow_m3hr / max(PI, 1e-9)
    return max(pbh, 0.0)

def run_nodal_analysis(params, flow_min=1.0, flow_max=400.0, n_points=200, tolerance_bar=3.0):
    segs = build_segments(params["trajectory"])
    flows = np.linspace(flow_min, flow_max, n_points)
    p_vlp = np.array([
        vlp(f, segs, params["rho"], params["mu"], 9.81, params["roughness"],
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

    return sol_flow, sol_pbh, sol_head, flows, p_vlp, p_ipr




try:
    from langchain_community.document_loaders import PyMuPDFLoader
except Exception:
    PyMuPDFLoader = None

try:
    from langchain_community.document_loaders import DocxLoader, CSVLoader, ImageLoader
except Exception:
    DocxLoader = None
    CSVLoader = None
    ImageLoader = None

# PDF support libs (tables/captions/ocr)
pdf_support = {"pdfplumber": None, "camelot": None, "fitz": None}
try:
    import pdfplumber
    pdf_support["pdfplumber"] = pdfplumber
except Exception:
    pass

try:
    import camelot
    pdf_support["camelot"] = camelot
except Exception:
    pass

try:
    import fitz  # PyMuPDF
    pdf_support["fitz"] = fitz
except Exception:
    pass

# Excel + pandas
excel_support = {"pandas": None}
try:
    import pandas as pd
    excel_support["pandas"] = pd
except Exception:
    pass





# OCR support (images + PDF raster)
ocr_support = {"pytesseract": None, "PIL": None}
try:
    import pytesseract
    from PIL import Image
    ocr_support["pytesseract"] = pytesseract
    ocr_support["PIL"] = Image
except Exception:
    pass

# Voice (optional; guarded)
try:
    import pyttsx3
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    if voices:
        engine.setProperty('voice', voices[0].id)
    def SpeakNow(text):
        engine.say(text)
        engine.runAndWait()
except Exception:
    def SpeakNow(text):
        pass

import speech_recognition as sr
from streamlit_pdf_viewer import pdf_viewer

# ----------------------------
# Folders
# ----------------------------
os.makedirs('pdfFiles', exist_ok=True)
os.makedirs('vectorDB', exist_ok=True)
os.makedirs('assets', exist_ok=True)

# ----------------------------
# Session state setup
# ----------------------------
if 'template' not in st.session_state:
    st.session_state.template = """You are a precise document assistant. Extract and synthesize insights across text, tables, figures, and charts. Quote exact values with units and reference table/figure labels or sheet names when available. If uncertainty exists, state it briefly and suggest checks.

Context:
{context}

History:
{history}

User:
{question}

Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question",
        chat_memory=StreamlitChatMessageHistory()
    )

# LLMs
if 'text_llm' not in st.session_state:
    st.session_state.text_llm = Ollama(
        base_url="http://localhost:11434",
        model="llama3",
        verbose=True
    )

# Embeddings
if 'text_embedder' not in st.session_state:
    st.session_state.text_embedder = OllamaEmbeddings(
        base_url="http://localhost:11434",
        model="llama3"
    )

# Vectorstore
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(
        persist_directory='vectorDB',
        embedding_function=st.session_state.text_embedder
    )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []



# ----------------------------
# Helpers: document model
# ----------------------------
from langchain_core.documents import Document

def df_to_markdown_and_csv(df, title=None):
    # Markdown
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = df.to_string(index=False)
    if title:
        md = f"Table: {title}\n\n{md}"
    # CSV
    csv_buf = StringIO()
    try:
        df.to_csv(csv_buf, index=False)
        csv_str = csv_buf.getvalue()
    finally:
        csv_buf.close()
    return md, csv_str

def extract_pdf_text_base(source):
    docs = []
    # Prefer PyMuPDFLoader for better layout, fallback to PyPDFLoader
    try:
        if PyMuPDFLoader is not None:
            base_loader = PyMuPDFLoader(source)
        else:
            base_loader = PyPDFLoader(source)
        base_docs = base_loader.load()
        for d in base_docs:
            d.metadata = {**d.metadata, "source": source, "content_type": "text"}
        docs.extend(base_docs)
    except Exception as e:
        st.warning(f"PDF base text load failed: {e}")
    return docs

def extract_pdf_tables_camelot(source):
    docs = []
    if not pdf_support["camelot"]:
        return docs
    # Try lattice first, then stream
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(source, pages='all', flavor=flavor)
            for i, t in enumerate(tables):
                df = t.df
                md, csv_str = df_to_markdown_and_csv(df, title=f"{Path(source).name} Table {i+1} ({flavor})")
                docs.append(Document(
                    page_content=md,
                    metadata={
                        "source": source, "content_type": "table",
                        "label": f"Table {i+1}", "parser": f"camelot-{flavor}", "table_csv": csv_str
                    }
                ))
        except Exception as e:
            # Continue to next flavor
            continue
    return docs

def extract_pdf_tables_pdfplumber(source):
    docs = []
    if not pdf_support["pdfplumber"]:
        return docs
    try:
        with pdf_support["pdfplumber"].open(source) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                # pdfplumber table extraction
                try:
                    tables = page.extract_tables()
                    for t_idx, table in enumerate(tables, start=1):
                        df = excel_support["pandas"].DataFrame(table) if excel_support["pandas"] else None
                        if df is not None:
                            md, csv_str = df_to_markdown_and_csv(df, title=f"{Path(source).name} p{page_idx} Table {t_idx} (pdfplumber)")
                        else:
                            # Fallback to simple formatting
                            md = f"Table on page {page_idx}, index {t_idx}:\n" + "\n".join([", ".join(row) for row in table])
                            csv_str = "\n".join([",".join(row) for row in table])
                        docs.append(Document(
                            page_content=md,
                            metadata={
                                "source": source, "content_type": "table",
                                "label": f"Table p{page_idx}-{t_idx}", "parser": "pdfplumber", "page": page_idx,
                                "table_csv": csv_str
                            }
                        ))
                except Exception:
                    pass
                # Figure/graph captions heuristic
                text = page.extract_text() or ""
                lines = text.splitlines()
                fig_blocks = []
                for i, line in enumerate(lines):
                    if re.search(r'\b(Figure|Fig\.)\s*\d+', line, flags=re.I):
                        block = [line]
                        for j in range(1, 3):
                            if i + j < len(lines):
                                block.append(lines[i + j])
                        fig_blocks.append("\n".join(block))
                for k, cap in enumerate(fig_blocks):
                    docs.append(Document(
                        page_content=f"Figure Caption (page {page_idx}):\n{cap}",
                        metadata={"source": source, "content_type": "figure_caption", "page": page_idx, "label": f"Figure {k+1}"}
                    ))
    except Exception as e:
        st.info(f"pdfplumber extraction skipped: {e}")
    return docs

def extract_pdf_ocr(source, dpi=200):
    docs = []
    # OCR scanned PDFs by rasterizing pages
    if not (pdf_support["fitz"] and ocr_support["pytesseract"] and ocr_support["PIL"]):
        return docs
    try:
        doc = pdf_support["fitz"].open(source)
        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            zoom = dpi / 72.0
            mat = pdf_support["fitz"].Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = ocr_support["PIL"].Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = ocr_support["pytesseract"].image_to_string(img)
            if text.strip():
                docs.append(Document(
                    page_content=f"OCR page {page_idx+1}:\n{text}",
                    metadata={"source": source, "content_type": "ocr_text", "page": page_idx+1, "parser": "pymupdf+tesseract"}
                ))
        doc.close()
    except Exception as e:
        st.info(f"PDF OCR skipped: {e}")
    return docs

def extract_pdf(file_path):
    source = str(file_path)
    docs = []
    docs.extend(extract_pdf_text_base(source))
    # Tables (Camelot + pdfplumber fallback)
    docs.extend(extract_pdf_tables_camelot(source))
    docs.extend(extract_pdf_tables_pdfplumber(source))
    # OCR (scanned PDFs)
    docs.extend(extract_pdf_ocr(source))
    return docs

def extract_excel(file_path):
    docs = []
    source = str(file_path)
    if not excel_support["pandas"]:
        st.error("pandas not available. Install pandas to parse Excel files.")
        return docs

    try:
        xls = pd.ExcelFile(source)
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            summary = f"Sheet '{sheet_name}' summary: {df.shape[0]} rows x {df.shape[1]} columns."
            docs.append(Document(
                page_content=summary,
                metadata={"source": source, "content_type": "sheet_summary", "sheet": sheet_name}
            ))
            md, csv_str = df_to_markdown_and_csv(df, title=f"{Path(source).name} - {sheet_name}")
            docs.append(Document(
                page_content=md,
                metadata={"source": source, "content_type": "table", "sheet": sheet_name, "table_csv": csv_str}
            ))
            # Numeric stats
            try:
                desc = df.select_dtypes(include=[np.number]).describe().transpose()
                stats_md, stats_csv = df_to_markdown_and_csv(desc, title=f"Descriptive stats - {sheet_name}")
                docs.append(Document(
                    page_content=stats_md,
                    metadata={"source": source, "content_type": "stats", "sheet": sheet_name, "table_csv": stats_csv}
                ))
            except Exception:
                pass
    except Exception as e:
        st.error(f"Excel parsing failed: {e}")
    return docs

def extract_image(file_path):
    """
    Full Vision Analysis + OCR + Table Extraction using Ollama (llava-phi3)
    Returns a Document with structured output.
    """

    import base64
    import requests
    from pathlib import Path
    from langchain_core.documents import Document

    source = str(file_path)

    # -----------------------------
    # Load image
    # -----------------------------
    try:
        with open(source, "rb") as f:
            img_bytes = f.read()
        b64 = base64.b64encode(img_bytes).decode()
    except Exception as e:
        return [
            Document(
                page_content=f"[Could not read image: {e}]",
                metadata={"source": source, "content_type": "image_meta"}
            )
        ]

    # -----------------------------
    # Vision prompt (FULL)
    # -----------------------------
    vision_prompt = (
        "You are an advanced Vision-OCR model.\n"
        "Your tasks:\n"
        "1) Describe the image in detail (objects, diagrams, graphs, engineering equipment).\n"
        "2) Extract ALL readable text exactly as in the image.\n"
        "3) Detect and reconstruct any tables.\n"
        "Respond ONLY in this structured format:\n\n"
        "DESCRIPTION:\n<your detailed visual description>\n\n"
        "TEXT OCR:\n<all extracted text>\n\n"
        "TABLES:\n<tables in plain text>\n"
    )

    # -----------------------------
    # Call llava-phi3 (Ollama)
    # -----------------------------
    OLLAMA_URL = "http://localhost:11434/api/generate"

    payload = {
        "model": "llava-phi3",
        "prompt": vision_prompt,
        "images": [b64]
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)

        # Handle old Ollama "streaming" text format
        try:
            data = r.json()
            vision_text = data.get("response", "")
        except:
            # fallback: remove streaming garbage
            vision_text = r.text.split('"}')[-1]

    except Exception as e:
        return [
            Document(
                page_content=f"[Vision model unavailable: {e}]",
                metadata={"source": source, "content_type": "image_meta"}
            )
        ]

    # -----------------------------
    # Return structured document
    # -----------------------------
    return [
        Document(
            page_content=f"VISION ANALYSIS OUTPUT for {Path(source).name}:\n\n{vision_text}",
            metadata={"source": source, "content_type": "image_vision"}
        )
    ]
import json
import requests

def parse_ollama_streaming_response(raw_text):
    """Aggregate 'response' fields from Ollama streaming NDJSON"""
    final_text = ""
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "response" in obj:
                final_text += obj["response"]
        except json.JSONDecodeError:
            continue
    return final_text.strip()

def ask_vision_question(vision_text, user_question, model="llava-phi3", timeout=300):
    """
    Send the user_question to the vision model using vision_text as context.
    Returns a cleaned textual answer (string).
    """
    if not vision_text:
        return "No vision context available to answer the question."

    OLLAMA_URL = "http://localhost:11434/api/generate"

    prompt = (
        "You are a helpful assistant. Use ONLY the following VISION ANALYSIS CONTEXT "
        "to answer the user's question. If the information is not present or ambiguous, "
        "explicitly say 'I cannot determine from the image.'\n\n"
        "VISION ANALYSIS CONTEXT:\n"
        f"{vision_text}\n\n"
        "USER QUESTION:\n"
        f"{user_question}\n\n"
        "Answer concisely and directly. Do NOT invent facts. If you must make a best-effort "
        "inference, label it as an inference."
    )

    payload = {
        "model": model,
        "prompt": prompt,
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        return f"Vision model request failed: {e}"

    # Parse JSON or fallback to streaming NDJSON
    answer = ""
    try:
        data = r.json()
        answer = data.get("response", "")
        if not answer:
            answer = parse_ollama_streaming_response(r.text)
    except Exception:
        answer = parse_ollama_streaming_response(r.text)

    return answer.strip() if answer else "I cannot determine from the image."

def load_documents(file_path):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(file_path)
    elif ext in [".xlsx", ".xls"]:
        return extract_excel(file_path)
    elif ext == ".csv":
        if CSVLoader is None:
            st.error("CSVLoader unavailable.")
            return []
        loader = CSVLoader(file_path)
        docs = loader.load()
        for d in docs:
            d.metadata = {**d.metadata, "source": file_path, "content_type": "table"}
        return docs
    elif ext == ".docx":
        if DocxLoader is None:
            st.error("DocxLoader unavailable.")
            return []
        loader = DocxLoader(file_path)
        docs = loader.load()
        for d in docs:
            d.metadata = {**d.metadata, "source": file_path, "content_type": "text"}
        return docs
    elif ext in [".png", ".jpg", ".jpeg"]:
        return extract_image(file_path)
    else:
        st.error("Unsupported file type.")
        return []

def chunk_documents(docs):
    # Smaller chunk for tables/stats; larger for prose
    text_splitter_text = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    text_splitter_table = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

    chunks = []
    for d in docs:
        splitter = text_splitter_table if d.metadata.get("content_type") in {"table", "stats"} else text_splitter_text
        split = splitter.split_documents([d])
        chunks.extend(split)
    return chunks

def render_sources(source_docs):
    if not source_docs:
        return
    with st.expander("Sources"):
        for i, doc in enumerate(source_docs, start=1):
            md = doc.metadata or {}
            lines = []
            lines.append(f"- Source: {md.get('source', 'unknown')}")
            if 'content_type' in md:
                lines.append(f"- Content type: {md['content_type']}")
            if 'sheet' in md:
                lines.append(f"- Sheet: {md['sheet']}")
            if 'page' in md:
                lines.append(f"- Page: {md['page']}")
            if 'label' in md:
                lines.append(f"- Label: {md['label']}")
            if 'parser' in md:
                lines.append(f"- Parser: {md['parser']}")
            st.markdown(f"**Source {i}:**")
            st.markdown("\n".join(lines))
            # Show a short preview
            preview = doc.page_content
            st.code(preview[:800] + ("..." if len(preview) > 800 else ""), language="markdown")

def try_plot_from_sources(source_docs):
    if not excel_support["pandas"]:
        st.info("Plotting requires pandas.")
        return
    plot_count = 0
    for doc in source_docs:
        md = doc.metadata or {}
        csv_str = md.get("table_csv")
        if not csv_str:
            continue
        try:
            df = pd.read_csv(StringIO(csv_str))
            # Basic heuristic: plot numeric columns against index
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 1:
                st.line_chart(df[numeric_cols])
                plot_count += 1
        except Exception:
            continue
    if plot_count == 0:
        st.info("No plottable tables found in retrieved sources.")
# ----------------------------
# Home tab
# ----------------------------
selected = option_menu(
    menu_title="Main Menu",
    options=["Home", "History", "Voice Text"],
    icons=["house", "book", "mic"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    key="main_menu"   # <-- unique key
)

# selected_new = option_menu(
#     menu_title="Analysis Menu",
#     options=[
#         "Summary Generation",
#         "Nodal Analysis Calculation",
#         "Nodal Analysis Results",
#         "Vision Part"
#     ],
#     icons=["list-task", "calculator", "file-earmark-pdf", "camera"],
#     menu_icon="cast",
#     default_index=0,
#     orientation="horizontal",
#     key="analysis_menu"   # <-- different unique key
# )

# ----------------------------
# Voice input
# ----------------------------
if selected == "Voice Text":
    with sr.Microphone() as source2:
        st.write("Listening for your question...")
        recog = sr.Recognizer()
        try:
            recog.adjust_for_ambient_noise(source2, duration=2)
            audio_data = recog.listen(source2, timeout=10, phrase_time_limit=15)
            try:
                text = recog.recognize_google(audio_data).lower()
                SpeakNow(f"Did you say: {text}")
                st.chat_input(text)
            except Exception as e:
                st.error(f"Voice recognition failed: {e}")
        except Exception as e:
            st.error(f"Microphone error: {e}")

# ----------------------------
# PDF history viewer
# ----------------------------
if selected == "History":
    pdf_folder = "pdfFiles"
    pdf_files = [f"{pdf_folder}/{f}" for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    if pdf_files:
        selected_pdf = st.selectbox("Select a PDF file", pdf_files)
        with open(selected_pdf, "rb") as f:
            binary_data = f.read()
            pdf_viewer(input=binary_data, width=700)
    else:
        st.info("No PDFs uploaded yet.")



if selected == "Home":
    st.title(f"You have selected {selected}")
    col1, col2 = st.columns(2)
    with col1:
        if Path("./assets/robot-lunch.gif").exists():
            st.image("./assets/robot-lunch.gif")
    with col2:
        st.title("What can I do for You?")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf","docx","csv","png","jpg","jpeg","xlsx","xls"]
    )

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

    if uploaded_file is not None:
        file_path = f'pdfFiles/{uploaded_file.name}'
        if not os.path.exists(file_path):
            st.info("Saving and processing the uploaded file...")
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.read())

            docs = load_documents(file_path)
            if docs:
                chunks = chunk_documents(docs)
                st.session_state.vectorstore.add_documents(chunks)
                st.session_state.vectorstore.persist()

        # Retriever
        st.session_state.retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 6}
        )

        # QA chain with source documents returned
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.text_llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            return_source_documents=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

        # Controls
        auto_plot = st.checkbox("Auto-plot numeric tables from retrieved sources", value=True)


##
from datetime import datetime
import re
from fpdf import FPDF

import re

import re
from datetime import datetime
from fpdf import FPDF




import re
from datetime import datetime
from pathlib import Path
from fpdf import FPDF

# def safe_text(text, max_len=400):
#     """Break long words and truncate text for safe PDF rendering."""
#     if not text:
#         return ""
#     text = str(text).replace("\n", " ")
#     # Insert spaces into very long words (every 80 chars)
#     safe = re.sub(r"(\S{80})(?=\S)", r"\1 ", text)
#     return safe[:max_len] + ("..." if len(text) > max_len else "")
#
# def save_response_to_pdf(answer_text, source_docs, filename="output.pdf", user_query=None):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#
#     # Timestamp + user query
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     header = f"Generated on: {timestamp}\n"
#     if user_query:
#         header += f"User Query: {user_query}\n"
#     safe_multicell(pdf, 0, 10, header, max_len=500)
#     pdf.ln(5)
#
#     # Answer section
#     safe_multicell(pdf, 0, 10, f"Answer:\n{answer_text}", max_len=1000)
#
#     # Sources section
#     pdf.set_font("Arial", 'B', 12)
#     pdf.cell(0, 10, "Supporting Sources:", ln=True)
#     pdf.set_font("Arial", size=10)
#
#     for i, doc in enumerate(source_docs, start=1):
#         md = getattr(doc, "metadata", {}) or {}
#         safe_multicell(pdf, 0, 8, f"Source {i}: {md.get('source', 'unknown')}", max_len=300)
#         preview = getattr(doc, "page_content", "")
#         safe_multicell(pdf, 0, 8, preview, max_len=400)
#         pdf.ln(5)
#
#         # Embed table if available
#         if md.get("table_csv"):
#             try:
#                 import pandas as pd
#                 from io import StringIO
#                 df = pd.read_csv(StringIO(md["table_csv"]))
#                 pdf.set_font("Arial", 'B', 10)
#                 pdf.cell(0, 8, "Table preview:", ln=True)
#                 pdf.set_font("Arial", size=8)
#                 for idx, row in df.head(5).iterrows():
#                     row_str = ", ".join(map(str, row.values))
#                     safe_multicell(pdf, 0, 6, row_str, max_len=200)
#                 pdf.ln(3)
#             except Exception:
#                 pass
#
#         # Embed image if available
#         if md.get("content_type") in {"image_meta", "image_ocr"}:
#             img_path = md.get("source")
#             if img_path and Path(img_path).exists():
#                 try:
#                     pdf.image(img_path, w=80)
#                     pdf.ln(5)
#                 except Exception:
#                     pass
#
#     pdf.output(filename)
#     return filename

# ----------------------------
# User chat input
# ----------------------------
if user_input := st.chat_input("Ask about tables, figures, trends, or calculations:"):
    st.session_state.chat_history.append({"role":"user","message":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain(user_input)

        st.session_state.chat_history.append({"role": "assistant", "message": response['result']})
        st.markdown(response['result'])

        # Sources
        source_docs = response.get('source_documents', [])
        render_sources(source_docs)

        # Save to PDF (with query + timestamp)
        pdf_file = f"pdfFiles/{Path(uploaded_file.name).stem}_answer.pdf"
        save_response_to_pdf(response['result'], source_docs, filename=pdf_file, user_query=user_input)

        # Download button
        with open(pdf_file, "rb") as f:
            st.download_button(
                label="📄 Download Answer PDF",
                data=f,
                file_name=Path(pdf_file).name,
                mime="application/pdf"
            )

        # Optional inline preview
        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()
            pdf_viewer(input=pdf_bytes, width=700)

import math
import numpy as np
import pandas as pd
import re
from io import StringIO

# Note: You must ensure pandas is imported globally as pd.

# Defaults used when RAG doesn't provide a value
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


def parse_scalar_from_text(text, key, unit_hint=None):
    """Heuristically extract scalar like 'Reservoir pressure = 230 bar'."""
    if not text:
        return None
    t = text.lower()
    patterns = []
    if key == "reservoir_pressure":
        patterns = [r"reservoir\s+pressure[^0-9]*([\d\.]+)"]
    elif key == "wellhead_pressure":
        patterns = [r"wellhead\s+pressure[^0-9]*([\d\.]+)"]
    elif key == "PI":
        patterns = [r"(productivity\s+index|pi)[^0-9]*([\d\.]+)"]
    elif key == "esp_depth":
        patterns = [r"(esp\s+(?:intake\s+)?depth)[^0-9]*([\d\.]+)"]
    elif key == "rho":
        patterns = [r"(density|rho)[^0-9]*([\d\.]+)"]
    elif key == "mu":
        patterns = [r"(viscosity|mu)[^0-9]*([\d\.]+)"]
    elif key == "roughness":
        patterns = [r"(roughness)[^0-9]*([\d\.eE\-]+)"]
    else:
        return None

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
    """Parse a pump curve from a CSV string with columns like Flow, Head."""
    try:
        df = pd.read_csv(pd.io.common.StringIO(csv_text))
        flow_col = next((c for c in df.columns if "flow" in c.lower()), None)
        head_col = next((c for c in df.columns if "head" in c.lower()), None)
        if flow_col and head_col:
            # Ensure safe conversion to float, dropping non-numeric rows
            flows = df[flow_col].apply(pd.to_numeric, errors='coerce').dropna().tolist()
            heads = df[head_col].apply(pd.to_numeric, errors='coerce').dropna().tolist()
            return {"flow": flows, "head": heads}
    except Exception:
        pass
    return None


def parse_trajectory_from_csv(csv_text):
    """Parse trajectory rows with columns MD, TVD, ID."""
    try:
        df = pd.read_csv(pd.io.common.StringIO(csv_text))
        md_col = next((c for c in df.columns if c.strip().upper() == "MD"), None)
        tvd_col = next((c for c in df.columns if c.strip().upper() == "TVD"), None)
        id_col = next((c for c in df.columns if c.strip().upper() in {"ID", "DIAMETER", "INNER DIAMETER"}), None)
        if md_col and tvd_col and id_col:
            rows = []
            # Safe conversion before appending
            df['MD_float'] = pd.to_numeric(df[md_col], errors='coerce')
            df['TVD_float'] = pd.to_numeric(df[tvd_col], errors='coerce')
            df['ID_float'] = pd.to_numeric(df[id_col], errors='coerce')
            df.dropna(subset=['MD_float', 'TVD_float', 'ID_float'], inplace=True)

            for _, r in df.iterrows():
                rows.append({"MD": r['MD_float'], "TVD": r['TVD_float'], "ID": r['ID_float']})

            if rows:
                rows = sorted(rows, key=lambda x: x["MD"])
                return rows
    except Exception:
        pass
    return None


def get_parameters_from_rag(source_docs):
    """Aggregate parameters from doc metadata and text. Fallback to DEFAULTS if not found."""
    params = {
        "rho": None, "mu": None, "roughness": None,
        "reservoir_pressure": None, "wellhead_pressure": None, "PI": None, "esp_depth": None,
        "pump_curve": None, "trajectory": None,
    }

    for doc in source_docs or []:
        md = getattr(doc, "metadata", {}) or {}
        text = getattr(doc, "page_content", "") or ""

        # Prefer explicit metadata if present
        for k in ["rho", "mu", "roughness", "reservoir_pressure", "wellhead_pressure", "PI", "esp_depth"]:
            if params[k] is None and md.get(k) is not None:
                try:
                    params[k] = float(md[k])
                except Exception:
                    pass

        # Parse pump curve
        if params["pump_curve"] is None:
            # ... (pump curve parsing logic) ...
            pass

        # Parse trajectory
        if params["trajectory"] is None:
            # ... (trajectory parsing logic) ...
            pass

        # Heuristic extraction from text
        for k in ["reservoir_pressure", "wellhead_pressure", "PI", "esp_depth", "rho", "mu", "roughness"]:
            if params[k] is None:
                val = parse_scalar_from_text(text, k)
                if val is not None:
                    params[k] = val

    # Apply defaults for anything still missing (CRITICAL STEP for NaN prevention)
    for k, v in DEFAULTS.items():
        if params.get(k) is None:
            params[k] = v
        # Ensure complex structures (like pump_curve/trajectory) always have a default structure
        elif k in ["pump_curve", "trajectory"] and not params[k]:
            params[k] = v

    return params


def build_segments(trajectory):
    """Build (L, D, theta) segments from MD/TVD/ID points."""
    segments = []
    # Ensure trajectory has at least 2 points for a segment
    if len(trajectory) < 2:
        return segments

    for i in range(1, len(trajectory)):
        MD = trajectory[i]["MD"] - trajectory[i - 1]["MD"]
        TVD = trajectory[i]["TVD"] - trajectory[i - 1]["TVD"]
        D = trajectory[i]["ID"]
        L = MD
        # Handle division by zero for straight vertical sections (MD=0)
        theta = math.atan2(TVD, MD if MD != 0 else 1e-9)
        segments.append((L, D, theta))
    return segments


def swamee_jain(Re, D, roughness):
    """Calculate Darcy friction factor using Swamee-Jain correlation."""
    if Re <= 0:
        return 0.0
    try:
        f = 0.25 / (math.log10((roughness / (3.7 * D)) + (5.74 / (Re ** 0.9)))) ** 2
        return f
    except (ValueError, ZeroDivisionError):
        return 0.0


def pump_interp(flow, pump_curve, key):
    """Interpolate head/power from the pump curve."""
    # Ensure curve arrays are non-empty and flows are sorted
    if not pump_curve["flow"] or not pump_curve[key] or len(pump_curve["flow"]) != len(pump_curve[key]):
        return 0.0
    return np.interp(flow, pump_curve["flow"], pump_curve[key])


def vlp(flow_m3hr, segments, rho, mu, g, roughness, esp_depth, pump_curve, wellhead_pressure):
    """Calculate Vertical Lift Performance (VLP) pressure drop."""
    q = flow_m3hr / 3600.0  # m3/hr to m3/s
    dp_total = 0.0
    depth_accum = 0.0

    for (L, D, theta) in segments:
        A = math.pi * D ** 2 / 4.0
        u = q / A if A > 0 else 0.0
        Re = rho * abs(u) * D / mu if mu > 0 else 0.0
        f = swamee_jain(Re, D, roughness)

        dp_fric = f * (L / max(D, 1e-9)) * (rho * u ** 2 / 2.0)
        dp_grav = rho * g * L * math.sin(theta)
        dp_total += dp_fric + dp_grav
        depth_accum += L * math.sin(theta)

    # Pump head adjustment
    if depth_accum >= esp_depth and pump_curve:
        # dp_total represents pressure required at bottom, pump adds head (reduces required pressure)
        dp_total -= rho * g * pump_interp(flow_m3hr, pump_curve, "head")

    return wellhead_pressure + dp_total / 1e5  # Pa to bar


def ipr(flow_m3hr, reservoir_pressure, PI):
    """Calculate Inflow Performance Relationship (IPR) pressure."""
    # Linear IPR model (simplified)
    # Check for valid PI to avoid division by zero
    if PI <= 1e-9:
        return reservoir_pressure

    pbh = reservoir_pressure - flow_m3hr / PI
    return max(pbh, 0.0)


def run_nodal_analysis(params, flow_min=1.0, flow_max=400.0, n_points=200, tolerance_bar=3.0):
    """Runs the main nodal analysis calculation."""
    # Pre-calculate segments (must handle potential issues in trajectory)
    segs = build_segments(params["trajectory"])

    # Check for minimal data integrity before running loops
    if not segs or params["PI"] <= 0 or not params["pump_curve"]["flow"] or len(params["pump_curve"]["flow"]) < 2:
        return None, None, None, np.array([]), np.array([]), np.array([])

    flows = np.linspace(flow_min, flow_max, n_points)

    # Calculate VLP and IPR arrays
    p_vlp = np.array([
        vlp(f, segs, params["rho"], params["mu"], 9.81, params["roughness"],
            params["esp_depth"], params["pump_curve"], params["wellhead_pressure"])
        for f in flows
    ])
    p_ipr = np.array([ipr(f, params["reservoir_pressure"], params["PI"]) for f in flows])

    diff = np.abs(p_vlp - p_ipr)
    idx = np.argmin(diff)

    # Find operating point
    if diff.size > 0 and diff[idx] < tolerance_bar:
        sol_flow = flows[idx]
        sol_pbh = p_vlp[idx]
        sol_head = pump_interp(sol_flow, params["pump_curve"], "head") if params["pump_curve"] else None
    else:
        sol_flow = sol_pbh = sol_head = None

    return sol_flow, sol_pbh, sol_head, flows, p_vlp, p_ipr


# Helper to ensure safety when reading extracted values in the UI (defined globally for use in both blocks)
def safe_float_value(extracted_data, key, default_value):
    """Safely converts an extracted parameter to float, using a default on failure."""
    try:
        value = extracted_data.get(key)
        if value is not None:
            return float(value)
        return float(default_value)
    except (TypeError, ValueError):
        return float(default_value)
# ----------------------------
# UI setup
# ----------------------------


def multipage_menu(caption):
    st.sidebar.title(caption)
    st.sidebar.subheader('Navigation')
    if Path('assets/ss.png').exists():
        st.sidebar.image('assets/ss.png')

# multipage_menu("LocoChat")
#
# selected = option_menu(
#     menu_title="Main Menu",
#     options=["Home", "History", "Voice Text"],
#     icons=["house", "book", "mic"],
#     menu_icon="cast",
#     default_index=0,
#     orientation="horizontal",
# )

# ----------------------------
# Additional UI for new features
# ----------------------------
st.header("Advanced Analysis Options")

selected_new = option_menu(
    menu_title="Analysis Menu",
    options=[
        "Summary Generation",
        "Nodal Analysis Calculation",
        "Nodal Analysis Results",
        "Vision Part"
    ],
    icons=["list-task", "calculator", "file-earmark-pdf", "camera"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)
import json
import base64
import requests
from pathlib import Path
from PIL import Image

def run_ollama_vision(image_path):
    import base64
    import requests

    url = "http://localhost:11434/api/generate"

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    payload = {
        "model": "llava-phi3",
        "prompt": "Extract all readable text from the image accurately.",
        "images": [base64.b64encode(image_bytes).decode("utf-8")]
    }

    resp = requests.post(url, json=payload, stream=True)

    full_response = ""

    # Ollama streams MULTIPLE JSON chunks → read line-by-line
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            obj = json.loads(line.decode("utf-8"))
            chunk = obj.get("response", "")
            full_response += chunk
        except:
            # Ignore partial chunks
            pass

    return full_response.strip()
# ----------------------------
# Option 1: Summary Generation
# ----------------------------
if selected_new == "Summary Generation":
    st.subheader("Generate a Summary")
    summary_length = st.selectbox("Select summary length:", ["Short", "Medium", "Long"])
    summary_type = st.selectbox("Select summary type:", ["Technical", "General", "Executive"])

    if user_input := st.chat_input("Enter text or ask for a summary:"):
        with st.spinner("Generating summary..."):
            # Use your QA chain or summarization logic
            response = st.session_state.qa_chain(user_input)
            summary_text = response['result']

        st.markdown(f"### {summary_length} {summary_type} Summary")
        st.markdown(summary_text)

        # Save summary to PDF
        pdf_file = f"pdfFiles/summary_{summary_length}_{summary_type}.pdf"
        save_response_to_pdf(summary_text, response.get('source_documents', []),
                             filename=pdf_file, user_query=user_input)

        with open(pdf_file, "rb") as f:
            st.download_button("📄 Download Summary PDF", f, file_name=Path(pdf_file).name, mime="application/pdf")


# ----------------------------
# Option 2: Nodal Analysis Calculation
# ----------------------------
# ----------------------------
# Option 2: Nodal Analysis Calculation
# ----------------------------
elif selected_new == "Nodal Analysis Calculation":
    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import datetime  # Required for PDF export

    st.subheader("Nodal Analysis (RAG-driven)")

    # 0) Acquire source_docs by actively retrieving them (avoids NameError: response)
    parameter_query = "reservoir pressure, wellhead pressure, PI, ESP depth, density, viscosity, roughness, pump curve table, well trajectory table"
    source_docs = []
    try:
        if 'vectorstore' in st.session_state:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
            source_docs = retriever.invoke(parameter_query)
    except Exception:
        pass  # Will rely entirely on DEFAULTS

    # 1) Retrieve parameters via RAG (will fall back to DEFAULTS if source_docs is empty/fail)
    extracted = get_parameters_from_rag(source_docs)

    # 2) Show extracted parameters and allow overrides
    st.markdown("### Parameters (auto-retrieved, editable)")
    col1, col2, col3 = st.columns(3)
    with col1:
        # FIX: Use safe_float_value to initialize inputs, preventing NaN if extraction failed
        rho = st.number_input("Density rho [kg/m3]", value=safe_float_value(extracted, "rho", DEFAULTS["rho"]))
        mu = st.number_input("Viscosity mu [Pa.s]", value=safe_float_value(extracted, "mu", DEFAULTS["mu"]),
                             format="%.6f")
        roughness = st.number_input("Pipe roughness [m]",
                                    value=safe_float_value(extracted, "roughness", DEFAULTS["roughness"]),
                                    format="%.6f")
    with col2:
        reservoir_pressure = st.number_input("Reservoir pressure [bar]",
                                             value=safe_float_value(extracted, "reservoir_pressure",
                                                                    DEFAULTS["reservoir_pressure"]))
        wellhead_pressure = st.number_input("Wellhead pressure [bar]",
                                            value=safe_float_value(extracted, "wellhead_pressure",
                                                                   DEFAULTS["wellhead_pressure"]))
        PI = st.number_input("Productivity Index [m3/hr per bar]",
                             value=safe_float_value(extracted, "PI", DEFAULTS["PI"]))
    with col3:
        esp_depth = st.number_input("ESP intake depth [m]",
                                    value=safe_float_value(extracted, "esp_depth", DEFAULTS["esp_depth"]))
        flow_min = st.number_input("Flow min [m3/hr]", value=1.0)
        flow_max = st.number_input("Flow max [m3/hr]", value=400.0)

    # Pump curve editable
    st.markdown("### Pump Curve (Flow vs Head)")
    default_flows = ",".join(map(lambda x: f"{x}", extracted["pump_curve"]["flow"]))
    default_heads = ",".join(map(lambda x: f"{x}", extracted["pump_curve"]["head"]))
    pump_flows_str = st.text_input("Flows (comma-separated)", default_flows)
    pump_heads_str = st.text_input("Heads (comma-separated)", default_heads)

    try:
        pump_curve = {
            "flow": [float(x) for x in pump_flows_str.split(",") if x.strip() != ""],
            "head": [float(x) for x in pump_heads_str.split(",") if x.strip() != ""],
        }
        if len(pump_curve["flow"]) != len(pump_curve["head"]) or len(pump_curve["flow"]) < 2:
            st.warning("Pump curve must have equal counts for flows and heads, with at least 2 points.")
    except ValueError:
        st.error("Invalid input in Pump Curve fields. Ensure inputs are comma-separated numbers.")
        pump_curve = extracted["pump_curve"]  # Fallback to extracted/default

    # Trajectory editable
    st.markdown("### Well Trajectory (MD, TVD, ID)")
    traj_df = pd.DataFrame(extracted["trajectory"])
    traj_df = st.data_editor(traj_df, num_rows="dynamic")
    trajectory = traj_df.to_dict(orient="records")
    trajectory = sorted(trajectory, key=lambda t: t["MD"])

    # 3) Run nodal analysis
    params = {
        "rho": rho, "mu": mu, "roughness": roughness,
        "reservoir_pressure": reservoir_pressure, "wellhead_pressure": wellhead_pressure,
        "PI": PI, "esp_depth": esp_depth, "pump_curve": pump_curve, "trajectory": trajectory
    }

    sol_flow, sol_pbh, sol_head, flows, p_vlp, p_ipr = run_nodal_analysis(
        params, flow_min=flow_min, flow_max=flow_max, n_points=200, tolerance_bar=3.0
    )

    # 4) Display results
    st.subheader("Results")
    if sol_flow:
        st.success(
            f"Operating point found:\n"
            f"- Flowrate: {sol_flow:.2f} m3/hr\n"
            f"- Bottomhole pressure (BHP): {sol_pbh:.2f} bar\n"
            f"- Pump head: {sol_head:.1f} m" if sol_head is not None else
            f"- Flowrate: {sol_flow:.2f} m3/hr\n- BHP: {sol_pbh:.2f} bar\n- Pump head: (not available)"
        )
    else:
        st.error("No solution found with current settings. Consider adjusting PI, pump curve, or trajectory.")

    # 5) Plot
    fig, ax = plt.subplots()
    ax.plot(flows, p_vlp, label="VLP")
    ax.plot(flows, p_ipr, label="IPR")
    if sol_flow:
        ax.scatter(sol_flow, sol_pbh, color="red", label="Operating point")
    ax.set_xlabel("Flowrate [m3/hr]")
    ax.set_ylabel("Pressure [bar]")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # 6) Optional: export a summary PDF
    if st.button("Export nodal analysis to PDF"):
        summary = (
            f"Nodal Analysis Summary\n\n"
            f"Density: {rho} kg/m3 | Viscosity: {mu} Pa.s | Roughness: {roughness} m\n"
            f"Reservoir pressure: {reservoir_pressure} bar | Wellhead pressure: {wellhead_pressure} bar | PI: {PI} m3/hr/bar\n"
            f"ESP depth: {esp_depth} m\n\n"
            f"Pump curve points: {len(pump_curve['flow'])}\n"
            f"Trajectory segments: {len(trajectory) - 1}\n\n"
            f"Operating point: "
            f"{'Q=' + f'{sol_flow:.2f} m3/hr, ' if sol_flow else ''}"
            f"{'BHP=' + f'{sol_pbh:.2f} bar, ' if sol_pbh else ''}"
            f"{'Head=' + f'{sol_head:.1f} m' if sol_head is not None else ''}"
        )
        pdf_file = f"nodal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        save_response_to_pdf(summary, source_docs, filename=pdf_file, user_query="Nodal Analysis")
        st.success(f"PDF saved: {pdf_file}")


############### 3 Nodal Analysis
# In your Streamlit app:
############### 3 Nodal Analysis Results
elif selected_new == "Nodal Analysis Results":
    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import datetime
    from pathlib import Path

    st.subheader("⛽ Nodal Analysis Results (RAG-driven Agent Workflow)")
    st.markdown("---")

    # 1. System Readiness Check
    if 'vectorstore' not in st.session_state or 'qa_chain' not in st.session_state:
        st.warning("Please upload and process a document in the 'Home' tab first to initialize the analysis system.")
        st.stop()

    # 2. Retrieve docs for nodal analysis parameters
    parameter_query = "reservoir pressure, wellhead pressure, PI, ESP depth, density, viscosity, roughness, pump curve table, well trajectory table"
    source_docs = []
    try:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
        source_docs = retriever.invoke(parameter_query)
    except Exception as e:
        st.error(f"Error retrieving documents for analysis: {e}")
        source_docs = []

    # 3–4. Extract + validate parameters
    params = get_parameters_from_rag(source_docs)

    # 5. Compute nodal analysis
    # Rely on defaults defined globally if specific values weren't retrieved/extracted
    sol_flow, sol_pbh, sol_head, flows, p_vlp, p_ipr = run_nodal_analysis(
        params, flow_min=1.0, flow_max=400.0, n_points=200, tolerance_bar=3.0
    )
    results = {
        "sol_flow": sol_flow, "sol_pbh": sol_pbh, "sol_head": sol_head,
        "flows": flows, "p_vlp": p_vlp, "p_ipr": p_ipr
    }

    # ----------------------------------------------------
    ## 📝 Results Summary Generation
    # ----------------------------------------------------

    summary_text = "Analysis complete. Data points extracted and calculation performed."

    if source_docs:
        summary_query = f"""
        Generate a concise, technical paragraph summary (5-8 sentences) for the Nodal Analysis. 
        Use only the context provided by the retrieved sources (tables/text). 
        Mention the key operating point found by the calculation: Q={results['sol_flow']:.2f} m³/hr, BHP={results['sol_pbh']:.2f} bar, Head={results['sol_head']:.1f} m (if available).
        Also, quote the most critical static parameter values from the source documents (e.g., Reservoir Pressure, PI, ESP Depth).
        """
        with st.spinner("Generating analytical summary from RAG context..."):
            try:
                # FIX: Use the 'query' key for the RetrievalQA chain invocation
                summary_response = st.session_state.qa_chain({"query": summary_query})
                summary_text = summary_response['result']
            except Exception as e:
                summary_text = f"Could not generate LLM summary: {e}. Showing default analysis text instead."

    st.markdown("## 📊 Analysis Report & Summary")
    st.info(summary_text)
    st.markdown("---")

    # ----------------------------------------------------
    ## 🔑 Key Operating Point
    # ----------------------------------------------------
    col_op, col_err = st.columns([2, 3])
    with col_op:
        st.markdown("### Operating Point")
        if results["sol_flow"]:
            st.success(
                f"**Flowrate (Q):** {results['sol_flow']:.2f} m³/hr\n\n"
                f"**Bottomhole Pressure (BHP):** {results['sol_pbh']:.2f} bar\n\n"
                f"**Pump Head:** {results['sol_head']:.1f} m" if results['sol_head'] is not None else
                f"**Bottomhole Pressure (BHP):** {results['sol_pbh']:.2f} bar"
            )
        else:
            st.error("❌ No stable operating point found within tolerance (±3.0 bar).")

    with col_err:
        st.markdown("### Extracted Fluid & Well Data")
        st.metric("Reservoir Pressure", f"{params['reservoir_pressure']} bar")
        st.metric("Productivity Index (PI)", f"{params['PI']} m³/hr/bar")
        st.metric("ESP Depth", f"{params['esp_depth']} m")

    st.markdown("---")

    # ----------------------------------------------------
    ## ⚙️ Detailed Parameters & Trajectory
    # ----------------------------------------------------
    st.markdown("## ⚙️ Detailed Extracted Data")

    col_pump, col_traj = st.columns(2)

    with col_pump:
        st.markdown("**Pump Curve (Flow vs. Head)**")
        if "pump_curve" in params:
            pump_df = pd.DataFrame({
                "Flow (m³/hr)": params["pump_curve"]["flow"],
                "Head (m)": params["pump_curve"]["head"]
            })
            st.dataframe(pump_df, use_container_width=True)
        else:
            st.warning("Pump curve data missing.")

    with col_traj:
        st.markdown("**Well Trajectory (MD, TVD, ID)**")
        if "trajectory" in params:
            traj_df = pd.DataFrame(params["trajectory"])
            st.dataframe(traj_df, use_container_width=True)
        else:
            st.warning("Trajectory data missing.")

    st.markdown("---")

    # ----------------------------------------------------
    ## 📈 Nodal Analysis Plot
    # ----------------------------------------------------
    st.markdown("## 📈 Nodal Analysis Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results["flows"], results["p_vlp"], label="VLP (Vertical Lift Performance)", color='blue', linewidth=2)
    ax.plot(results["flows"], results["p_ipr"], label="IPR (Inflow Performance Relationship)", color='green',
            linewidth=2)

    if results["sol_flow"]:
        ax.scatter(results["sol_flow"], results["sol_pbh"], color="red", s=100, zorder=5, label="Operating Point")
        ax.axvline(results["sol_flow"], color='red', linestyle='--', linewidth=0.5)
        ax.axhline(results["sol_pbh"], color='red', linestyle='--', linewidth=0.5)

    ax.set_xlabel("Flowrate [m³/hr]", fontsize=12)
    ax.set_ylabel("Pressure [bar]", fontsize=12)
    ax.set_title("Nodal Analysis (IPR vs VLP)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig)

    st.markdown("---")

    # ----------------------------------------------------
    ## 📑 Sources & Export
    # ----------------------------------------------------

    render_sources(source_docs)

    # 7. Export to PDF with proper formatting
    if st.button("Export Full Results to PDF"):

        try:
            # ----------------------------------------------------
            # Build Operating Point Table
            # ----------------------------------------------------
            op_table = {
                "Flowrate (m³/hr)": f"{results['sol_flow']:.2f}",
                "Bottomhole Pressure (bar)": f"{results['sol_pbh']:.2f}",
                "Pump Head (m)": f"{results['sol_head']:.1f}",
            }

            # ----------------------------------------------------
            # Build Parameter Table (Automatic Dict Table Builder)
            # ----------------------------------------------------
            parameter_dict = {
                "Reservoir Pressure (bar)": params.get("reservoir_pressure", "-"),
                "Wellhead Pressure (bar)": params.get("wellhead_pressure", "-"),
                "Productivity Index (m³/hr/bar)": params.get("PI", "-"),
                "ESP Depth (m)": params.get("esp_depth", "-"),
                "Fluid Density (kg/m³)": params.get("density", "-"),
                "Viscosity (cP)": params.get("viscosity", "-"),
                "Tubing Roughness (mm)": params.get("roughness", "-"),
            }

            # ----------------------------------------------------
            # SAVE nodal plot image temporarily
            # ----------------------------------------------------
            plot_path = "temp_nodal_plot.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")

            # ----------------------------------------------------
            # Build summary text
            # ----------------------------------------------------
            main_analysis_text = f"""
            Nodal Analysis Summary

            Operating Point:
            Flowrate (Q): {results['sol_flow']:.2f} m³/hr
            Bottomhole Pressure (BHP): {results['sol_pbh']:.2f} bar
            Pump Head: {results['sol_head']:.1f} m

            Extracted Parameters:
            Reservoir Pressure: {params.get('reservoir_pressure', '-')} bar
            PI: {params.get('PI', '-')} m³/hr/bar
            ESP Depth: {params.get('esp_depth', '-')} m

            Full LLM Summary:
            {summary_text}
            """

            # ----------------------------------------------------
            # CALL UPDATED PDF GENERATOR
            # ----------------------------------------------------
            pdf_file = f"nodal_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

            save_response_to_pdf(
                answer_text=main_analysis_text,
                source_docs=source_docs,
                filename=pdf_file,
                user_query="Nodal Analysis Results",
                operating_point=op_table,  # >>> ADDED <<<
                parameter_dict=parameter_dict,  # >>> ADDED <<<
                nodal_plot_path=plot_path  # >>> ADDED <<<
            )

            st.success(f"📄 PDF saved: {pdf_file}")

            with open(pdf_file, "rb") as f:
                st.download_button("⬇️ Download Results PDF", f, file_name=pdf_file, mime="application/pdf")

        except Exception as e:
            st.error(f"Failed to generate or save PDF: {e}")

# ----------------------------
# Option 4: Vision Part
# ----------------------------


# ----------------------------
# Option 4: Vision Part
# ----------------------------

# -----------------------------
# Helper: Extract text from image using Ollama Vision Model
# -----------------------------
import requests
import json
from pathlib import Path
import base64

# -----------------------------
# Extract text + description from image
# -----------------------------
import base64
import requests
import json
from pathlib import Path

def extract_image(file_path, model="llava-phi3", timeout=120):
    """Extract readable text and description from an image using the vision model."""
    try:
        with open(file_path, "rb") as f:
            img_bytes = f.read()
        b64_image = base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        return f"[Error reading image: {e}]"

    prompt = (
        "You are an advanced vision-OCR model. "
        "1) Describe the image in detail (what it contains, objects, diagrams, equipment).\n"
        "2) Extract ALL readable text exactly.\n"
        "3) If tables exist, reconstruct them in plain text.\n"
        "Respond in clean structured format:\n\n"
        "DESCRIPTION:\n...\n\n"
        "TEXT OCR:\n...\n\n"
        "TABLES:\n...\n"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64_image]
    }

    try:
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        return f"[Vision model unavailable: {e}]"

    # Parse streaming JSON or normal JSON
    extracted_text = ""
    for line in r.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            extracted_text += obj.get("response", "")
        except:
            continue

    return extracted_text.strip() or "[No text detected]"

def ask_vision_question(vision_text, user_question, model="llava-phi3", timeout=120):
    if not vision_text:
        return "No vision context available to answer the question."

    prompt = (
        "You are a helpful assistant. Use ONLY the following VISION ANALYSIS CONTEXT "
        "to answer the user's question. If the information is not present or ambiguous, "
        "explicitly say 'I cannot determine from the image.'\n\n"
        "VISION ANALYSIS CONTEXT:\n"
        f"{vision_text}\n\n"
        "USER QUESTION:\n"
        f"{user_question}\n\n"
        "Answer concisely and directly. Do NOT invent facts. "
        "If you must make a best-effort inference, label it as an inference."
    )

    payload = {
        "model": model,
        "prompt": prompt
    }

    try:
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        return f"[Vision model request failed: {e}]"

    # parse streaming JSON
    answer = ""
    for line in r.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            answer += obj.get("response", "")
        except:
            continue

    return answer.strip() or "I cannot determine from the image."


# -----------------------------
# Helper: Save PDF
# -----------------------------
def save_response_to_pdf(answer_text, filename="vision_output.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.multi_cell(0, 10, "VISION ANALYSIS OUTPUT", align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, answer_text)
    pdf.output(filename)
    return filename

# -----------------------------
# Vision Part UI
# -----------------------------
st.subheader("Vision-based Chat & OCR Analysis")

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    # show image
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    # save temporarily
    img_path = f"pdfFiles/{uploaded_image.name}"
    with open(img_path, "wb") as f:
        f.write(uploaded_image.read())

    # extract vision text
    vision_text = extract_image(img_path)

    st.markdown("### 🔍 Vision Analysis Output")
    st.text_area("Vision Analysis", vision_text, height=250)

    # initialize chat history
    if "vision_chat" not in st.session_state:
        st.session_state["vision_chat"] = []

    # Display chat
    for msg in st.session_state["vision_chat"]:
        if msg["role"] == "user":
            st.markdown(f"**👤 You:** {msg['content']}")
        else:
            st.markdown(f"**🧠 Assistant:** {msg['content']}")

    # input box
    user_input = st.text_input(
        "Ask the assistant about the image (e.g. 'What is this image about?')",
        key="vision_input",
        value=""
    )
    send = st.button("Send")

    if send and user_input:
        st.session_state["vision_chat"].append({"role": "user", "content": user_input})

        with st.spinner("Generating answer from vision model..."):
            reply = ask_vision_question(vision_text, user_input)

        st.session_state["vision_chat"].append({"role": "assistant", "content": reply})

        # refresh UI
        st.experimental_rerun()

    # Always provide PDF download
    pdf_file = f"pdfFiles/vision_{Path(uploaded_image.name).stem}.pdf"
    save_response_to_pdf(vision_text, pdf_file)
    with open(pdf_file, "rb") as f:
        st.download_button("📄 Download Vision PDF", f, file_name=Path(pdf_file).name, mime="application/pdf")

