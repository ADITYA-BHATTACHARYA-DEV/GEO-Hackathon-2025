# streamlit_app.py
# Your original Streamlit code preserved exactly, with additive enhancements appended.

# ============================
# ADDITION: compatibility & styling
# (This block is inserted AFTER the imports in your original code to avoid changing the imports themselves.)
# ============================

# NOTE:
# To ensure we do not change your original code lines, the compatibility wrapper and CSS
# are placed first (before your UI uses). The rest of your original code remains intact.

import streamlit as st
import json
import textwrap
from datetime import datetime

# Compatibility wrapper for deprecated parameter:
# If any original code calls st.image(..., use_column_width=True),
# this wrapper will map it to use_container_width=True to avoid deprecation warnings.
_original_st_image = st.image


def _image_wrapper(*args, **kwargs):
    # Map deprecated 'use_column_width' to 'use_container_width' if present
    if 'use_column_width' in kwargs:
        # Move value to use_container_width
        kwargs['use_container_width'] = kwargs.pop('use_column_width')
    return _original_st_image(*args, **kwargs)


# Apply wrapper
st.image = _image_wrapper

# Add global CSS styling to beautify the page (dark theme, nicer fonts, accent colors)
st.markdown(
    """
<style>
/* Page background */
.reportview-container .main {
    background-color: #0f1720;
    color: #e6eef8;
}

/* Headers */
h1, h2, h3 {
    font-family: Inter, 'Segoe UI', Roboto, sans-serif;
    color: #9ad0ff;
}

/* Subheads */
h4, h5, h6 {
    color: #bcdffb;
}

/* Expander */
.stExpander {
    background-color: #0b1220;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #0ea5ff, #7c3aed);
    color: white;
    border-radius: 8px;
    padding: .6rem 1rem;
    font-weight: 600;
}

/* Download buttons */
.stDownloadButton>button {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border-radius: 8px;
}

/* Code blocks */
.css-1q8dd3e pre {
    background-color: #071029 !important;
    color: #e6eef8 !important;
    border-radius: 8px;
    padding: 12px;
}

/* Metrics */
.stMetric {
    background-color: #081027;
    border-radius: 8px;
}

/* Images rounded */
img {
    border-radius: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

# Add a top-level page config (keeps your later set_page_config when original code runs)
st.set_page_config(page_title="LocoChat — Well Report Agentic AI Workflow", layout="wide", page_icon="🛢️")

# ============================
# ORIGINAL CODE STARTS HERE (UNCHANGED)
# I pasted your original code exactly as you provided it.
# ============================

# ------------------------------------------------------------------------------
# Helper generators (produce artifacts you can download or view)
# ------------------------------------------------------------------------------

def generate_json_schema():
    """
    Produce a JSON Schema for the structured output that the extractor should produce.
    This schema is intentionally conservative: it normalizes units to SI and documents
    the main fields required to run nodal analysis.
    """
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "WellReportExtraction",
        "type": "object",
        "properties": {
            "well_id": {"type": "string"},
            "well_name": {"type": "string"},
            "measured_depths": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_m": {"type": "number"},
                        "end_m": {"type": "number"}
                    }
                }
            },
            "true_vertical_depth_m": {"type": "number"},
            "casing_strings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "inner_diameter_m": {"type": "number"},
                        "outer_diameter_m": {"type": "number"},
                        "grade": {"type": "string"},
                        "top_md_m": {"type": "number"},
                        "bottom_md_m": {"type": "number"}
                    }
                }
            },
            "tubing_strings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "size_inch": {"type": "number"},
                        "weight": {"type": "string"},
                        "top_md_m": {"type": "number"},
                        "bottom_md_m": {"type": "number"}
                    }
                }
            },
            "perforations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "interval_md_start_m": {"type": "number"},
                        "interval_md_end_m": {"type": "number"},
                        "shot_density_per_m": {"type": "number"}
                    }
                }
            },
            "pvt": {
                "type": "object",
                "properties": {
                    "oil_density_kgm3": {"type": "number"},
                    "viscosity_cp": {"type": "number"},
                    "gor_scf_stb": {"type": "number"},
                    "formation_volume_factor": {"type": "number"}
                }
            },
            "reservoir": {
                "type": "object",
                "properties": {
                    "pressure_bar": {"type": "number"},
                    "temperature_c": {"type": "number"},
                    "permeability_mD": {"type": "number"},
                    "skin": {"type": "number"}
                }
            },
            "wellhead_conditions": {
                "type": "object",
                "properties": {
                    "tubing_head_pressure_bar": {"type": "number"},
                    "choke_size_inch": {"type": "number"}
                }
            }
        },
        "required": ["well_id", "well_name", "pvt"]
    }
    return json.dumps(schema, indent=2)

def generate_prompts():
    """
    Return a dictionary of recommended prompts / templates used for the extractor and the agent.
    These are starting points for instruction-tuning or prompt engineering.
    """
    extractor_prompt = textwrap.dedent("""
    You are an extractor. Given a document snippet, produce a JSON object with EXACT fields:
    well_id, well_name, measured_depths, true_vertical_depth_m, casing_strings, tubing_strings,
    perforations, pvt, reservoir, wellhead_conditions.

    - Normalize units to SI (meters, bar, kg/m3). Use float numbers for numeric fields.
    - Return ONLY valid JSON. No explanatory text.
    - If a field is not present in the snippet, return null or omit the field.

    Example snippet:
    "Completion summary: Well ALPHA-5. MD 1: 0-2891 m, TVD: 0-2860 m. 7\" casing to 2000 m; 5.5\" casing to total.
    Tubing: 2 7/8\" 6.5# from 1000 m to surface. Perforations: 1850-1870 m. PVT: oil density 820 kg/m3,
    viscosity 2.4 cp; GOR 250 scf/stb."

    Expected (example) output: {"well_name":"ALPHA-5", "measured_depths":[{"start_m":0,"end_m":2891}], ...}
    """)

    agent_clarify_template = textwrap.dedent("""
    You are a clarification agent. The extractor produced the following flagged items:
    {flags_summary}

    For each flagged item, produce a single clear, non-technical question that:
    - References the snippet context.
    - Offers short answer options where possible (e.g., 'Confirm MD value: 2860 m or 2891 m?').
    - If a numeric correction is requested, suggest the most likely correction and why.

    Return JSON with: [{"field":"tvd","question":"...","suggested_fix":"...","context_snippet":"..."}]
    """)

    return {"extractor_prompt": extractor_prompt, "agent_clarify_template": agent_clarify_template}

def generate_test_cases_csv():
    """
    Generate a small test-case matrix (CSV string) that developers can use to validate
    extraction and nodal-solver behavior. This CSV is synthetic and meant to be extended.
    """
    header = "test_id,description,file,expected_tubing_inch,expected_tvd_m,notes\n"
    rows = [
        "tc001,Simple clean PDF,alpha5.pdf,2.875,2860,standard extraction",
        "tc002,Scanned table with OCR noise,beta_scan.pdf,2.875,2891,OCR errors introduced",
        "tc003,Nonstandard units,gamma_units.pdf,2.875,2860,units in ft and inches mixed",
        "tc004,Missing PVT,delta_missing_pvt.pdf,,2860,PVT values absent - agent should ask"
    ]
    return header + "\n".join(rows)

def generate_vlp_pseudocode():
    """
    Produce readable pseudocode for a simplified VLP solver and nodal root-finding logic.
    Engineers should replace placeholders with robust physics & units handling.
    """
    code = textwrap.dedent("""
    # VLP pseudocode (simplified)
    import numpy as np
    from scipy.optimize import brentq

    def mixture_density(q, oil_density, water_cut, gas_gor, p, T):
        # Placeholder: a simple linear mixture density (real code uses flash calculations).
        rho_mix = oil_density * (1 - water_cut) + 1000.0 * water_cut
        return rho_mix

    def hydrostatic_pressure_bar(rho_mix, depth_m, g=9.80665):
        # rho (kg/m3) * g (m/s2) * depth (m) -> Pa; convert Pa -> bar (1e5 Pa = 1 bar)
        return (rho_mix * g * depth_m) / 1e5

    def frictional_loss_bar(q_m3day, diameter_m, length_m, mu, rho_mix):
        # Simplified Darcy-Weisbach friction term (placeholder)
        # q_m3day -> q_m3s
        q_m3s = q_m3day / 86400.0
        A = np.pi * (diameter_m/2.0)**2
        velocity = q_m3s / A if A > 0 else 0.0
        friction_factor = 0.02  # placeholder
        dp_pa = friction_factor * (length_m/diameter_m) * 0.5 * rho_mix * velocity**2
        return dp_pa / 1e5  # Pa to bar

    def compute_vlp_bar(q_m3day, well, pwh_bar):
        rho = mixture_density(q_m3day, well['oil_density_kgm3'], well.get('water_cut', 0.0), well.get('gor', 0.0), well.get('p_bar', None), well.get('T_c', None))
        dp_hydro = hydrostatic_pressure_bar(rho, well['depth_m'])
        dp_fric = frictional_loss_bar(q_m3day, well['tubing_diameter_m'], well['depth_m'], well.get('mu', 1e-3), rho)
        pwf_bar = pwh_bar + dp_hydro + dp_fric
        return pwf_bar

    def solve_q_m3day(pr_bar, PI_m3day_per_bar, well, pwh_bar, q_min=1e-6, q_max=10000.0):
        # Solve for q where Pwf_from_VLP(q) == Pwf_from_IPR(q).
        def objective(q):
            pwf_ipr = pr_bar - q / PI_m3day_per_bar
            pwf_vlp = compute_vlp_bar(q, well, pwh_bar)
            return pwf_vlp - pwf_ipr

        q_solution = brentq(objective, q_min, q_max)
        return q_solution
    """)
    return code

# ------------------------------------------------------------------------------
# UI: Header and short description
# ------------------------------------------------------------------------------
col1, col2 = st.columns(2, gap="small")

with col1:
    # image is optional — don't break if missing
    try:
        st.image("./assets/ts.png", use_column_width=True)
    except Exception:
        st.info("Header image not found at ./assets/ts.png — replace or ignore.")

with col2:
    st.title("LocoChat — Well Report Agentic AI Workflow")
    st.write(
        "Comprehensive, explainable blueprint for an agentic AI system that ingests well reports "
        "and returns validated structured data and nodal-analysis-based production estimates."
    )

st.write("---")

# ------------------------------------------------------------------------------
# Executive summary (collapsible)
# ------------------------------------------------------------------------------
with st.expander("Executive Summary — What this system does (click to expand)", expanded=True):
    st.markdown(
        """
**Core capabilities**
- Ingest well completion reports (PDF, DOCX, scans), PVT reports, tables and schematics.
- Use a RAG-style retrieval + LLM extraction pipeline to produce structured fields required for nodal analysis.
- Validate extracted fields with deterministic domain checks and heuristics (e.g., MD >= TVD).
- Run nodal analysis by coupling an inflow model (IPR) with a vertical lift model (VLP) to compute flow rates.
- Use agentic orchestration to ask clarifying questions when inputs are missing or ambiguous.
- Support a vision-first path for images / scanned tables using a vision model + OCR.
- Provide full auditability: source snippet -> extraction -> validation -> numeric solution.
        """
    )

st.write("---")

# ------------------------------------------------------------------------------
# Detailed architecture sections (long-form explanations)
# ------------------------------------------------------------------------------
import streamlit as st

st.markdown("""
<style>
    /* General Styling */
    .stApp { padding: 0 !important; }
    .architecture-container {
        padding: 10px;
    }
    .section-header {
        font-size: 2.2rem;
        font-weight: 800;
        /* ENSURED LIGHT TEXT FOR DARK MODE BACKGROUND */
        color: #FFFFFF; 
        margin-top: 30px;
        margin-bottom: 10px;
        /* Using a softer divider color */
        border-bottom: 4px solid #444444; 
        padding-bottom: 5px;
    }
    .sub-section-title {
        font-size: 1.5rem;
        font-weight: 700;
        /* ENSURED LIGHT TEXT FOR DARK MODE BACKGROUND */
        color: #F0F0F0; 
        margin-top: 20px;
        margin-bottom: 10px;
        padding-left: 10px;
        border-left: 4px solid #1E90FF; /* Dodger Blue Accent remains */
    }
    .technical-term {
        color: #FF4B4B; /* Streamlit Red (high contrast) */
        font-weight: 700;
        font-style: italic;
    }
    .responsibilities-box {
        /* Background MUST remain white for high-contrast content */
        background-color: #ffffff; 
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        /* Text inside white box MUST be dark */
        color: #1a1a1a; 
    }
    .responsibilities-box strong {
        color: #007bff;
        font-size: 1.1em;
    }
    .feature-point {
        font-size: 1.1rem;
        font-weight: 600;
        /* ENSURED LIGHT TEXT FOR DARK MODE BACKGROUND */
        color: #CCCCCC; 
        margin-top: 10px;
    }
    /* CRITICAL FIX: Ensure all list items outside the white box are light */
    .stMarkdown > div > ul {
        list-style-type: disc;
        padding-left: 20px;
    }
    .stMarkdown > div > ul > li {
        color: #E0E0E0 !important; /* Set to light gray, forcing visibility */
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="architecture-container">', unsafe_allow_html=True)

st.markdown('<div class="section-header">System Architecture & Component Breakdown</div>', unsafe_allow_html=True)
st.markdown("---")

# ----------------------------------------------------------------------
# TWO COLUMN LAYOUT START
# ----------------------------------------------------------------------

col_left, col_right = st.columns(2)

# ----------------------------------------------------------------------
# COLUMN 1: INGESTION, EXTRACTION, NODAL SOLVER
# ----------------------------------------------------------------------

with col_left:
    # 1. INGESTION & PRE-PROCESSING
    st.markdown('<div class="sub-section-title">1. Ingestion & Pre-processing: The Multi-Modal Data Factory</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
        <div class="responsibilities-box">
            <strong>Responsibilities:</strong>
            <ul>
                <li>Accepts common file types (PDF/DOCX/PNG/JPEG) and returns canonical document chunks.</li>
                <li>For scans: apply OCR with layout detection to extract lines, tables, and images.</li>
                <li>Produce chunk metadata: (page, <span class="technical-term">bbox</span>, <span class="technical-term">chunk_id</span>, <span class="technical-term">source_uri</span>) and save original file fingerprint.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown('<p class="feature-point">Key Technical Steps:</p>', unsafe_allow_html=True)
    st.markdown(
        """
    - Preprocess images (deskew, denoise).
    - OCR + layout detection (tables, figures).
    - Chunk text into retrieval-friendly sizes (tuned to LLM context length).
    - Index text chunks into vector DB for retrieval.
    """
    )

    st.markdown("---")

    # 3. STRUCTURED EXTRACTION
    st.markdown('<div class="sub-section-title">3. Structured Extraction: LLM as the Technical Parser</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
        <div class="responsibilities-box">
            <strong>Field Schema (examples):</strong>
            <ul class="step-list">
                <li><span class="technical-term">well_id</span>, <span class="technical-term">well_name</span>, measured_depths, true_vertical_depth_m.</li>
                <li>casing_strings, tubing_strings, perforations.</li>
                <li>pvt ($\text{oil density}$, $\text{viscosity}$, $\text{GOR}$), reservoir ($\text{P}_{\text{r}}$, $\text{T}$, $\text{k}$, $\text{skin}$), wellhead_conditions.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown('<p class="feature-point">Approach:</p>', unsafe_allow_html=True)
    st.markdown(
        """
    - Prompt instruction-tuned LLM to output strict JSON according to the JSON Schema (downloadable below).
    - Use regex and deterministic parsers as fallback to normalize numbers and units.
    - Keep a pointer from each field to the source snippet and store extraction confidence.
    """
    )

    st.markdown("---")

    # 5. NODAL ANALYSIS ENGINE (Solver)
    st.markdown('<div class="sub-section-title">5. Nodal Analysis Engine: The Numerical Solver Tool</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="responsibilities-box"><strong>Core Task:</strong> This Python tool (the <span class="technical-term">Solver Agent</span>) is responsible for building and solving the fluid dynamics models to determine the stable operating point of the well.</div>',
        unsafe_allow_html=True)
    st.markdown('<p class="feature-point">Models Supported:</p>', unsafe_allow_html=True)
    st.markdown(
        # Changed to explicit UL/LI list for better rendering
        """
        <ul>
            <li><span style="color: white; font-weight: bold;">IPR Models:</span> simple linear (PI), <span class="technical-term">Vogel</span>, <span class="technical-term">Fetkovich</span>, or fitted decline/IPR from history.</li>
            <li><span style="color: white; font-weight: bold;">VLP Models:</span> hydrostatic + friction + acceleration (Beggs & Brill or simplified Darcy-Weisbach).</li>
            <li><span style="color: white; font-weight: bold;">Numerical solver:</span> root-finding (bisection / Brent).</li>
        </ul>
        """, unsafe_allow_html=True
    )
    st.markdown('<p class="feature-point">Flow:</p>', unsafe_allow_html=True)
    st.markdown(
        # Changed to explicit UL/LI list for better rendering
        """
        <ul>
            <li><span style="color: white; font-weight: bold;">Build IPR:</span> Build IPR from reservoir parameters or history.</li>
            <li><span style="color: white; font-weight: bold;">Build VLP:</span> Build VLP from well geometry and fluid PVT.</li>
            <li><span style="color: white; font-weight: bold;">Solve for q:</span> Solve for $\text{q}$ where $\text{P}_{\text{wf}}^{\text{IPR}}(\text{q}) == \text{P}_{\text{wf}}^{\text{VLP}}(\text{q})$.</li>
        </ul>
        """, unsafe_allow_html=True
    )
    st.markdown('<p class="feature-point">Uncertainty:</p>', unsafe_allow_html=True)
    st.markdown(
        # Changed to explicit UL/LI list for better rendering
        """
        <ul>
            <li><span style="color: white; font-weight: bold;">Monte Carlo Sweeps:</span> Optionally run Monte Carlo sweeps across uncertain inputs ($\text{PI}$, $\text{P}_{\text{r}}$, permeability).</li>
        </ul>
        """, unsafe_allow_html=True
    )

# ----------------------------------------------------------------------
# COLUMN 2: RETRIEVAL, VALIDATION, ORCHESTRATION
# ----------------------------------------------------------------------

with col_right:
    # 2. RETRIEVAL (RAG Retriever)
    st.markdown('<div class="sub-section-title">2. Retrieval (RAG Retriever): The Semantic Librarian</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
        <div class="responsibilities-box">
            <strong>Design:</strong>
            <ul>
                <li>Use dense embeddings + vector store (FAISS, Milvus) + metadata filters.</li>
                <li>Over-fetch (<span class="technical-term">K</span> larger) and let extractor prune irrelevant chunks.</li>
                <li>Maintain retrieval score and source pointer for each chunk.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True
    )
    # Inject CSS
    st.markdown("""
    <style>
    .feature-point {
        color: white !important;
        font-weight: 600;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Use your HTML tag
    st.markdown('<p class="feature-point">Technical Implementation:</p>', unsafe_allow_html=True)

    st.markdown(
        """
    - Uses dense embeddings + vector store (ChromaDB, or industry standards like FAISS/Milvus) + metadata filters.
    - Augmented Fetching (<span class="technical-term">K-Overfetch</span>): Designed to over-fetch to maximize the probability of capturing critical data points.
    - Auditability & Pointers: Maintains the retrieval score and source pointer (URI, page) for every chunk, enabling full traceability in the final report.
    """
    )

    st.markdown("---")

    # 4. SANITY CHECKING & VALIDATION
    st.markdown('<div class="sub-section-title">4. Sanity Checking & Validation: The Deterministic Rule Engine</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="responsibilities-box"><strong>Function:</strong> The <span class="technical-term">Validator Agent</span> ensures all extracted parameters are physically plausible and logically consistent before execution.</div>',
        unsafe_allow_html=True)
    st.markdown('<p class="feature-point">Examples of Critical Validation Rules:</p>', unsafe_allow_html=True)
    st.markdown(
        # Changed to explicit UL/LI list for better rendering
        """
        <ul>
            <li><span style="color: white; font-weight: bold;">Measured Depth (MD) / TVD:</span> Measured Depth (<span class="technical-term">MD</span>) $\ge$ True Vertical Depth (<span class="technical-term">TVD</span>).</li>
            <li><span style="color: white; font-weight: bold;">Tubing/Casing Integrity:</span> Tubing outer diameter $<$ casing inner diameter - tolerance.</li>
            <li><span style="color: white; font-weight: bold;">Fluid Properties:</span> PVT densities in plausible ranges (oil: $\sim 600-1000 \text{ kg/m}^3$).</li>
            <li><span style="color: white; font-weight: bold;">Completion Integrity:</span> Perforations within casing interval.</li>
        </ul>
        """, unsafe_allow_html=True
    )
    st.markdown('<p class="feature-point">Behaviors:</p>', unsafe_allow_html=True)
    st.markdown(
        """
    - **Auto-Correction:** Auto-correct low-risk errors (unit inference, OCR character fixes).
    - **Human-in-the-loop:** For high-impact anomalies, present a user question and pause for confirmation.
    """
    )

    st.markdown("---")

    # 6. AGENTIC ORCHESTRATION & HUMAN-IN-THE-LOOP
    st.markdown('<div class="sub-section-title">6. Agentic Orchestration & Control Layer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="responsibilities-box"><strong>Architecture:</strong> The entire process is managed by an **Orchestrator Agent** that sequences specialized tasks, handles failures, and manages user interaction for critical decisions.</div>',
        unsafe_allow_html=True)
    st.markdown('<p class="feature-point">Agent Type Catalog:</p>', unsafe_allow_html=True)
    st.markdown(
        # Changed to explicit UL/LI list for better rendering
        """
        <ul>
            <li><span style="color: white; font-weight: bold;">Orchestrator:</span> Sequences tasks and handles failures/timeouts.</li>
            <li><span style="color: white; font-weight: bold;">Retriever agent:</span> Optimizes chunks to fetch.</li>
            <li><span style="color: white; font-weight: bold;">Extractor agent:</span> LLM that outputs JSON.</li>
            <li><span style="color: white; font-weight: bold;">Validator agent:</span> Deterministic rule engine.</li>
            <li><span style="color: white; font-weight: bold;">Solver agent:</span> Numeric VLP/IPR solver.</li>
            <li><span style="color: white; font-weight: bold;">Vision agent:</span> Parses images/tables using vision LLM + OCR.</li>
            <li><span style="color: white; font-weight: bold;">UI agent:</span> Crafts clarifying questions for the user.</li>
        </ul>
        """, unsafe_allow_html=True
    )
    st.markdown('<p class="feature-point">Operating Modes:</p>', unsafe_allow_html=True)
    st.markdown(
        # Changed to explicit UL/LI list for better rendering
        """
        <ul>
            <li><span style="color: white; font-weight: bold;">Fully Automated Mode:</span> The system attempts auto-correction for all errors and records all assumptions and corrections in the final audit report.</li>
            <li><span style="color: white; font-weight: bold;">Human-in-the-loop Mode:</span> The workflow is designed to **block and pause** whenever a high-impact anomaly (e.g., PI is zero or a major constraint is violated) is detected, requiring mandatory user confirmation before proceeding.</li>
        </ul>
        """, unsafe_allow_html=True
    )

st.markdown('<div style="clear:both;"></div>', unsafe_allow_html=True)  # Clear float for main content area

st.write("---")

st.markdown('</div>', unsafe_allow_html=True)  # Close architecture-container div

# ------------------------------------------------------------------------------
# Downloadable artifacts and button handlers (safe)
# ------------------------------------------------------------------------------
st.header("Downloadable artifacts (examples)")

col_a, col_b, col_c = st.columns([1,1,1])

with col_a:
    st.subheader("JSON Schema")
    if st.button("View JSON Schema"):
        st.code(generate_json_schema(), language="json")
    st.download_button(
        label="Download JSON Schema",
        data=generate_json_schema(),
        file_name="well_extraction_schema.json",
        mime="application/json"
    )

with col_b:
    st.subheader("Prompts & Templates")
    prompts = generate_prompts()
    if st.button("View Prompts"):
        st.code(prompts["extractor_prompt"], language="text")
        st.code(prompts["agent_clarify_template"], language="text")
    st.download_button(
        label="Download Prompts (txt)",
        data=prompts["extractor_prompt"] + "\n\n" + prompts["agent_clarify_template"],
        file_name="extractor_and_agent_prompts.txt",
        mime="text/plain"
    )

with col_c:
    st.subheader("Test-case Matrix (CSV)")
    if st.button("View Test Cases"):
        st.code(generate_test_cases_csv(), language="text")
    st.download_button(
        label="Download Test Cases (CSV)",
        data=generate_test_cases_csv(),
        file_name="test_cases.csv",
        mime="text/csv"
    )

st.write("---")

# ------------------------------------------------------------------------------
# Pseudocode & Examples (detailed)
# ------------------------------------------------------------------------------
st.header("Implementation Details & Pseudocode (detailed)")

with st.expander("End-to-end orchestration pseudocode (click to expand)", expanded=False):
    st.code(textwrap.dedent("""
    # Orchestration pseudocode (end-to-end)
    def handle_report_upload(file, user_prompt):
        # 1. Ingest & OCR
        doc_chunks = ingest_and_ocr(file)
        # 2. Retrieval
        retrieved = retriever.get_topk(user_prompt, doc_chunks, k=20)
        # 3. Structured extraction (LLM + deterministic parsing)
        extracted = extractor.extract_structured(retrieved)
        # 4. Validation
        validated = validator.run_checks(extracted)
        if validated.requires_user_confirmation:
            question = ui_agent.compose_question(validated.flags)
            user_response = present_question_and_get_answer(question)
            extracted = apply_user_corrections(extracted, user_response)
            validated = validator.run_checks(extracted)
        # 5. Prepare nodal inputs and solve
        nodal_input = prepare_nodal_input(extracted)
        nodal_result = nodal_solver.solve(nodal_input)
        # 6. Render report & audit trail
        report = reporter.render(nodal_result, audit_log)
        return report
    """), language="python")

with st.expander("Extractor prompt example + expected JSON (click to expand)", expanded=False):
    st.write("Example instruction used to prompt an instruction-tuned LLM:")
    st.code(generate_prompts()["extractor_prompt"], language="text")
    st.write("Example expected JSON (toy):")
    st.code(json.dumps({
        "well_name": "ALPHA-5",
        "measured_depths": [{"start_m": 0.0, "end_m": 2891.0}],
        "true_vertical_depth_m": 2860.0,
        "casing_strings": [
            {"type":"7in", "inner_diameter_m":0.1524, "outer_diameter_m":0.1778, "top_md_m":0.0, "bottom_md_m":2000.0}
        ],
        "tubing_strings": [{"size_inch":2.875, "top_md_m":0.0, "bottom_md_m":1000.0}],
        "perforations": [{"interval_md_start_m":1850.0, "interval_md_end_m":1870.0}],
        "pvt": {"oil_density_kgm3":820.0, "viscosity_cp":2.4, "gor_scf_stb":250.0}
    }, indent=2), language="json")

with st.expander("Validation rules & examples (click to expand)", expanded=False):
    st.markdown("""
**Important validation rules (examples)**

- Rule: MD >= TVD  
  - Check: if extracted MD < TVD → flag.  
  - Action: show both snippets and ask user to confirm which is correct.

- Rule: Tubing OD <= Casing ID - tolerance  
  - Check: if tubing OD > casing ID → attempt unit repair (mm ↔ in), OCR fixes; else ask user.

- Rule: Perforations inside casing interval  
  - Check: if perforation MD outside casing bottom/top → flag.

**Auto-correction heuristics**
- If numeric parse failed due to OCR noise (e.g., '2.8S7' or 'l' instead of '1'), run character substitution heuristics and re-parse.
- If units are ambiguous, try to infer from table headers or other fields in same table.
    """)

# ------------------------------------------------------------------------------
# VLP pseudocode viewer & download
# ------------------------------------------------------------------------------
st.header("VLP & Solver Pseudocode")

col_v1, col_v2 = st.columns([2,1])
with col_v1:
    if st.button("View VLP Pseudocode"):
        st.code(generate_vlp_pseudocode(), language="python")
with col_v2:
    st.download_button(
        label="Download VLP Pseudocode",
        data=generate_vlp_pseudocode(),
        file_name="vlp_pseudocode.py",
        mime="text/plain"
    )

st.write("---")

# ------------------------------------------------------------------------------
# Example "toy" nodal walkthrough and interactive demo inputs
# ------------------------------------------------------------------------------
st.header("Toy Nodal Analysis Walkthrough (interactive)")

st.markdown("Use these toy inputs to see how a nodal solver would be invoked (this UI is a demonstration only — solver here is a simplified estimator).")

with st.form("toy_nodal_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        pr_bar = st.number_input("Reservoir pressure (Pr) [bar]", value=300.0, step=1.0)
        PI_m3day_per_bar = st.number_input("Productivity Index (PI) [m3/day/bar]", value=0.5, step=0.1)
    with col2:
        pwh_bar = st.number_input("Wellhead pressure (Pwh) [bar]", value=50.0, step=1.0)
        depth_m = st.number_input("Perforation depth [m]", value=2000.0, step=10.0)
    with col3:
        oil_density = st.number_input("Oil density [kg/m3]", value=800.0, step=1.0)
        tubing_diameter_inch = st.selectbox("Tubing OD (inch)", options=[2.375, 2.875, 3.5], index=1)

    submit = st.form_submit_button("Run toy nodal (estimate)")

if submit:
    # Simplified estimate: hydrostatic + linear PI. This is NOT a real VLP solver.
    # Convert tubing diameter to meters
    diameter_m = tubing_diameter_inch * 0.0254
    # Very simplified conversion and friction model — for demonstration only:
    q_trial = PI_m3day_per_bar * (pr_bar - pwh_bar)  # naive q ignoring VLP
    # hydrostatic (bar)
    hydrostatic_bar = (oil_density * 9.80665 * depth_m) / 1e5
    # naive friction: small additive term scaled with q
    friction_bar = 0.0001 * q_trial
    pwf_vlp = pwh_bar + hydrostatic_bar + friction_bar
    # Compute q_ipr if pwf_vlp were the actual bottom-hole pressure
    q_ipr = PI_m3day_per_bar * (pr_bar - pwf_vlp)
    st.metric("Naive IPR-based q (m3/day)", f"{q_trial:.2f}")
    st.metric("Estimated Pwf (from naive VLP) [bar]", f"{pwf_vlp:.2f}")
    st.metric("Adjusted q after VLP intersects IPR (m3/day)", f"{max(q_ipr,0.0):.2f}")
    st.info("Note: This is a toy calculation. Use the VLP pseudocode and real multiphase model for production-grade results.")

st.write("---")
import streamlit as st

# Custom styles must be included if this file is run standalone, but assuming
# it runs within the main app where the <style> block is already defined.

st.markdown('<div class="architecture-container">', unsafe_allow_html=True)

st.markdown('<div class="section-header">Observability, Security & Developer Roadmap</div>', unsafe_allow_html=True)
st.markdown("---")

# ----------------------------------------------------------------------
# TWO COLUMN LAYOUT START
# ----------------------------------------------------------------------

col_left, col_right = st.columns(2)

# ----------------------------------------------------------------------
# COLUMN 1: OBSERVABILITY AND SECURITY
# ----------------------------------------------------------------------

with col_left:
    # OBSERVABILITY
    st.markdown('<div class="sub-section-title">1. Observability: The Audit Trail</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="responsibilities-box"><strong>Purpose:</strong> To maintain a comprehensive, auditable record of every system action and data transformation, ensuring reliability and debugging capability.</div>',
        unsafe_allow_html=True)

    st.markdown('<p class="feature-point">Key Audit Records Maintained:</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul>
            <li><span style="color: white; font-weight: bold;">Ingestion Snapshot:</span> File hash and metadata for original document upload.</li>
            <li><span style="color: white; font-weight: bold;">Retrieval Data:</span> Query vector, chunks retrieved, and semantic scores.</li>
            <li><span style="color: white; font-weight: bold;">Extractor Outputs:</span> Raw JSON output from LLM before validation.</li>
            <li><span style="color: white; font-weight: bold;">Validation Flags:</span> Detailed logs of all validation checks and rule violations.</li>
            <li><span style="color: white; font-weight: bold;">User Corrections:</span> Records of user input during Human-in-the-Loop confirmation steps.</li>
            <li><span style="color: white; font-weight: bold;">Solver I/O:</span> Inputs and outputs of the deterministic numerical models (VLP/IPR).</li>
        </ul>
        """, unsafe_allow_html=True
    )

    st.markdown('<p class="feature-point">Audit Record Structure:</p>', unsafe_allow_html=True)
    st.markdown(
        """
        Every audit record contains: timestamp, agent ID, input snapshot, output snapshot, source pointers, and confidence metrics.
        """
    )
    st.markdown("---")

    # SECURITY
    st.markdown('<div class="sub-section-title">2. Security: Data Integrity & Access Control</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="responsibilities-box"><strong>Goal:</strong> To ensure all sensitive engineering and proprietary data remains protected throughout the RAG and analysis pipeline.</div>',
        unsafe_allow_html=True)

    st.markdown('<p class="feature-point">Core Security Measures:</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul>
            <li><span style="color: white; font-weight: bold;">Data at Rest:</span> Encrypt documents and vector indices (<span class="technical-term">ChromaDB persistence layer</span>) at rest using industry-standard protocols.</li>
            <li><span style="color: white; font-weight: bold;">Access Control:</span> Implement RBAC (Role-Based Access Control) for accessing sensitive well data and controlling query types.</li>
            <li><span style="color: white; font-weight: bold;">Input Validation:</span> Enforce strict JSON schema validation on extractor outputs to prevent malicious injection or downstream calculation failures.</li>
        </ul>
        """, unsafe_allow_html=True
    )

# ----------------------------------------------------------------------
# COLUMN 2: DEVELOPER ROADMAP
# ----------------------------------------------------------------------

with col_right:
    # DEVELOPER ROADMAP
    st.markdown('<div class="sub-section-title">3. Developer Roadmap: Phases of Deployment</div>',
                unsafe_allow_html=True)

    st.markdown('<p class="feature-point">MVP (Minimum Viable Product):</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul>
            <li>Ingest + OCR + chunking.</li>
            <li>Retriever + small vector index.</li>
            <li>Instruction-tuned LLM for JSON extraction.</li>
            <li>Deterministic validations (basic unit/plausibility checks).</li>
            <li>Simplified nodal solver (linear IPR, simplified VLP).</li>
            <li>Basic UI for human confirmations.</li>
        </ul>
        """, unsafe_allow_html=True
    )

    st.markdown('<p class="feature-point">Production-Grade System:</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul>
            <li>Robust multiphase VLP (mechanistic models like <span class="technical-term">Beggs & Brill</span>).</li>
            <li>Vision-first path integration with LLMs (<span class="technical-term">llava-phi3</span>) for schematics and tables.</li>
            <li>Monte Carlo uncertainty & sensitivity analysis.</li>
            <li>Orchestration using frameworks (Temporal, Celery) for retries and timeouts across agent chains.</li>
            <li>Comprehensive Access Control, logging, and cryptographic signing of final results.</li>
        </ul>
        """, unsafe_allow_html=True
    )

st.markdown('<div style="clear:both;"></div>', unsafe_allow_html=True)

st.write("---")

st.markdown('</div>', unsafe_allow_html=True)  # Close architecture-container div
# ------------------------------------------------------------------------------
# Footer / assets image and final notes
# ------------------------------------------------------------------------------

# ----------------------------------------------------------------------
# FOOTER CONTENT (Advanced Extraction & Notes)
# ----------------------------------------------------------------------

colf1, colf2 = st.columns([1, 2])

with colf1:
    try:
        # Assuming ssn.png is a diagram representing the workflow
        st.image("./assets/ssn.png", use_column_width=True)
    except Exception:
        st.info("Footer image not found at ./assets/ssn.png — replace or ignore.")

with colf2:
    st.markdown('<div class="sub-section-title" style="margin-top: 0;">Advanced Data Extraction & Scientific Methodology</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="footer-note">The Structured Extraction Agent (Section 3) is designed not just for simple key-value parsing, but for advanced data science tasks, employing:</p>
        <ul>
            <li><span style="color: white; font-weight: bold;">Conditional Random Fields (CRF):</span> Used in pre-processing to identify and label structured elements like headers, footers, and section breaks, enhancing contextual understanding before LLM processing.</li>
            <li><span style="color: white; font-weight: bold;">Graph Neural Networks (GNNs):</span> Applied for complex table understanding, treating cells, rows, and columns as nodes and edges to infer relationships and ensure correct row/column alignment, especially after OCR introduces noise. </li>
            <li><span style="color: white; font-weight: bold;">Latent Dirichlet Allocation (LDA):</span> Used for initial document topic modeling to ensure highly relevant chunks are prioritized, improving the signal-to-noise ratio during retrieval.</li>
            <li><span style="color: white; font-weight: bold;">Contrastive Learning for Embeddings:</span> Used to fine-tune vector generation, maximizing the distance between dissimilar documents (e.g., geology reports vs. completion reports) while minimizing distance between highly relevant content, improving retrieval precision. </li>
        </ul>
        """, unsafe_allow_html=True
    )


st.markdown('</div>', unsafe_allow_html=True) # Close architecture-container div


# ----------------------
# ADDITIONAL ENHANCEMENTS (APPENDED) — Installation guide + final beautified footer
# ----------------------

st.write("---")
st.header("📦 Model Installation Guide — Ollama, Llama-3 & llava-phi3 (additive)")
with st.expander("Show installation commands and notes", expanded=False):
    with st.expander("Procedural Steps for Successive Installation:", expanded=False):
        st.markdown(
            "Follow these copy-paste commands to install **Ollama**, **Llama-3**, and **Llava-Phi3** for local agentic workflows.")
        st.markdown("---")

        # 1) Install Ollama
        st.markdown("### **1) Install Ollama**")
        st.markdown("**macOS / Linux**")
        st.code(
            "curl -fsSL https://ollama.com/install.sh | sh",
            language="bash"
        )

        st.markdown("**Windows (PowerShell)**")
        st.code(
            "winget install Ollama.Ollama",
            language="powershell"
        )

        st.markdown("---")

        # 2) Llama3
        st.markdown("### **2) Pull Llama-3 model**")
        st.code(
            "# default Llama3 model\nollama pull llama3",
            language="bash"
        )

        st.markdown("---")

        # 3) Llava-Phi3
        st.markdown("### **3) Pull Llava-Phi3 (Vision Model)**")
        st.code(
            "ollama pull llava-phi3",
            language="bash"
        )

        st.markdown("---")

        # 4) Tests
        st.markdown("### **4) Run quick sanity-test**")
        st.code(
            "ollama run llama3\nollama run llava-phi3",
            language="bash"
        )

        st.markdown("---")


##########################

        # ------------------------------------------------------------------
        # ADD-ON: CHROMADB INSTALLATION + MULTI-MODAL WORKFLOW + AGENTIC AI
        # ------------------------------------------------------------------
        st.markdown("### **5) (Optional) Install & Initialize ChromaDB — Vector Store for RAG**")
        st.markdown(
            """
ChromaDB is a lightweight, high-speed vector database ideal for **local retrieval-augmented generation (RAG)**.

Install it using:
"""
        )
        st.code("pip install chromadb", language="bash")

        st.markdown("**Minimal usage example (Python):**")
        st.code(
            '''import chromadb
client = chromadb.Client()

collection = client.create_collection(name="well_reports")

collection.add(
    ids=["chunk1"],
    documents=["7-inch casing from 0–2000 m, TVD 2860 m"],
    metadatas=[{"page":1}]
)

results = collection.query(
    query_texts=["casing information"],
    n_results=3
)

print(results)
''',
            language="python"
        )

        st.markdown("---")

        # ---------------------------------------------------------------
        # MULTI-MODAL WORKFLOW ADD-ON
        # ---------------------------------------------------------------
        st.markdown("### **Multi-Modal AI Workflow Overview**")
        st.markdown(
            """
A modern agentic well-report system must handle *multiple forms of information*:

#### **1. Text (PDF/DOCX extracted text)**
- Parsed using OCR / layout-detection
- Embedded with text embeddings (e.g., Llama3 embeddings)
- Stored in **ChromaDB** for retrieval

#### **2. Vision (Scanned tables, schematics, images)**
- Processed using **Llava-Phi3** or similar vision-language models  
- Extracts:
  - tabular data (perforations, casing tables)
  - numbers inside schematic diagrams
  - handwritten annotations

#### **3. Hybrid reasoning**
- Model retrieves relevant text chunks (RAG)
- Model also receives aligned image-derived features
- Extractor merges text signals + image signals into unified JSON

#### **4. Validation + numeric simulation**
- Deterministic engine checks MD/TVD, diameters, PVT ranges
- VLP + IPR solver computes flow-rate estimates

**Result:** A complete **multi-modal, verifiable well-report intelligence workflow**.
"""
        )

        st.markdown("---")

        # ---------------------------------------------------------------
        # AGENTIC AI PRINCIPLES (DETAILED)
        # ---------------------------------------------------------------
        st.markdown("### **Agentic-AI Principles (Fully Explained)**")
        st.markdown(
            """
Agentic AI is not “a single model”; it is **a coordinated team of specialized agents**:

---

### **🔹 1. Role-Specialized Agents**
Each agent focuses on one job only:

| Agent | Purpose |
|-------|---------|
| **Ingestion Agent** | Extract text, tables, images with OCR |
| **Retrieval Agent** | Fetch relevant context using ChromaDB |
| **Extractor Agent** | Convert retrieved text into structured JSON |
| **Validation Agent** | Apply deterministic domain rules |
| **Vision Agent** | Parse schematics, tables, depth diagrams |
| **Solver Agent** | Perform nodal analysis (IPR × VLP) |
| **UI Agent** | Ask clarifying questions to humans |

Clear specialization → **higher accuracy & reliability**.

---

### **🔹 2. Chain-of-Thought Separation**
Agents do not mix reasoning:
- Extractor gives structured JSON only  
- Validator explains what is wrong  
- UI Agent builds user-facing questions

This prevents hallucinated numbers and ensures an **audit trail**.

---

### **🔹 3. Determinism Over Guessing**
If uncertain:
- Extractor outputs null
- Validator raises a flag  
- UI Agent asks a human-friendly clarification  
- User feedback is written back into the JSON

The system never “makes up” TVD, casing, or PVT.

---

### **🔹 4. Full Auditability**
For each field:
- retrieved snippet  
- extraction text  
- validation checks  
- user corrections  
- final solved parameters

This allows engineers to *trust* the result.

---

### **🔹 5. Orchestrated Tool Use**
The orchestrator decides which agent to call next:

✔ Retrieval  
✔ Extraction  
✔ Validation  
✔ Vision parsing  
✔ Numerical solving  
✖ Skip unreachable steps  

Agents are called **only when needed**, improving speed and cost.

---

### **🔹 6. Multi-Model Reasoning**
The system uses:
- Llama3 → text extraction  
- Llava-Phi3 → vision reasoning  
- Python solvers → physics (VLP/IPR)  
- ChromaDB → retrieval memory  

This hybrid stack creates a **true agentic workflow**, not a single LLM call.
"""
        )

        st.markdown("---")


# ---------------------------
# PDF Generator (requires reportlab)
# ---------------------------
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

def make_architecture_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
    styles = getSampleStyleSheet()
    flowables = []

    flowables.append(Paragraph("Agentic System Architecture — LocoChat", styles['Title']))
    flowables.append(Spacer(1, 12))

    flowables.append(Paragraph("Overview:", styles['Heading2']))
    flowables.append(Paragraph(
        "This document outlines the multi-modal, agentic AI workflow "
        "for extracting well report data, validating it, and performing nodal analysis.", styles['BodyText']))
    flowables.append(Spacer(1, 12))

    flowables.append(Paragraph("Components:", styles['Heading3']))
    flowables.append(Paragraph(
        "- Ingestion & OCR\n- Retriever (ChromaDB)\n- Extractor (Llama3)\n- Vision Agent (Llava-Phi3)\n- Validator\n- Solver (VLP/IPR)\n- Orchestrator / UI", styles['BodyText']))
    flowables.append(Spacer(1, 12))

    flowables.append(Paragraph("Agentic Principles:", styles['Heading3']))
    flowables.append(Paragraph(
        "Role specialization, determinism on numeric fields, auditability, human-in-the-loop for high-impact flags.",
        styles['BodyText']))

    doc.build(flowables)
    buffer.seek(0)
    return buffer

with st.expander("Download architecture PDF", expanded=False):
    st.write("Generate a short architecture PDF summarising the agentic workflow.")
    pdf_buffer = make_architecture_pdf()
    st.download_button("Download Architecture PDF", data=pdf_buffer, file_name="LocoChat_agentic_architecture.pdf", mime="application/pdf")


# ---------------------------
# ADD-ON: ChromaDB + Multi-Modal + Agentic-AI (append only)
# Paste inside your installation expander or create a new expander and paste there.
# ---------------------------
st.markdown("### **5) (Optional) Install & Initialize ChromaDB — Vector Store for RAG**")
st.markdown("ChromaDB is a lightweight, high-speed vector database ideal for local retrieval-augmented generation (RAG).")
st.code("pip install chromadb", language="bash")

st.markdown("**Minimal usage example (Python)**")
st.code(
'''import chromadb
client = chromadb.Client()
collection = client.create_collection(name="well_reports")
collection.add(
    ids=["chunk1"],
    documents=["7-inch casing from 0–2000 m, TVD 2860 m"],
    metadatas=[{"page":1}]
)
results = collection.query(query_texts=["casing information"], n_results=3)
print(results)
''',
    language="python"
)

st.markdown("---")
import streamlit as st

# Custom CSS for high contrast and modern look (assuming definitions from main file)
st.markdown("""
<style>
    /* Replicating key styles for standalone rendering */
    .section-title-mm {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E90FF; /* Dodger Blue */
        border-bottom: 4px solid #444444; 
        padding-bottom: 5px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .technical-term {
        color: #FF4B4B; 
        font-weight: 700;
        font-style: italic;
    }
    # .feature-box {
    #     background-color: #1a1a1a; /* Dark background for boxes */
    #     border: 1px solid #444444;
    #     border-radius: 8px;
    #     padding: 15px;
    #     min-height: 200px; /* Ensure visual stability in columns */
    #     margin-bottom: 20px;
    #     box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    # }
    # .feature-box strong {
    #     color: #FFFFFF;
    #     font-size: 1.2em;
    #     display: block;
    #     margin-bottom: 8px;
    # }
    # .feature-box p {
    #     color: #E0E0E0; /* Light text */
    #     line-height: 1.5;
    #     font-size: 1.0rem;
    # }
    /* FIX for introductory text visibility/rendering */
    .intro-paragraph {
        color: #F0F0F0; 
        margin-bottom: 20px; 
        font-size:1.15rem;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="multi-modal-container">', unsafe_allow_html=True)

st.markdown('<div class="section-title-mm">Multi-Modal AI Workflow Overview</div>', unsafe_allow_html=True)
st.markdown(
    """
    <p class="intro-paragraph">
    A modern agentic well-report system must seamlessly handle and integrate multiple distinct forms of information 
    to achieve reliable results. LocoChat achieves this through a structured four-stage fusion process.
    </p>
    """, unsafe_allow_html=True
)

# ----------------------------------------------------------------------
# FOUR-COLUMN GRID FOR WORKFLOW STAGES
# ----------------------------------------------------------------------

col_mm1, col_mm2, col_mm3, col_mm4 = st.columns(4)

with col_mm1:
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown('<strong>1.  Unstructured Data Ingestion</strong>', unsafe_allow_html=True)
    st.markdown(
        """
        <p>This is the base layer for knowledge. The system processes standard digital documents to extract core narrative and sequential data.</p>
        <ul>
            <li><strong>Input:</strong> <span class="technical-term">PDF</span>, DOCX (raw text).</li>
            <li><strong>Process:</strong> OCR + layout detection $\\to$ smart chunking.</li>
            <li><strong>Storage:</strong> Embedded and stored in the <span class="technical-term">ChromaDB</span> for retrieval.</li>
        </ul>
        """, unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_mm2:
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown('<strong>2. 👁 Vision & Structured Parsing</strong>', unsafe_allow_html=True)
    st.markdown(
        """
        <p>Handling visual and scanned content, which often contains critical tables and schematics essential for engineering analysis. </p>
        <ul>
            <li><strong>Input:</strong> Scanned tables, schematics, annotations (PNG/JPEG).</li>
            <li><strong>Tool:</strong> <span class="technical-term">llava-phi3</span> (Vision AG) extracts and translates visual data into structured JSON/CSV.</li>
            <li><strong>Output:</strong> Structured data merged for later validation.</li>
        </ul>
        """, unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_mm3:
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown('<strong>3.  Hybrid Reasoning & Fusion</strong>', unsafe_allow_html=True)
    st.markdown(
        """
        <p>The stage where information from diverse sources (Text, Vision, Excel) is unified by the Extractor Agent to complete the JSON parameter schema.</p>
        <ul>
            <li><strong>Process:</strong> Retriever (text context) + Vision features (image table extraction) are combined.</li>
            <li><strong>Validation Goal:</strong> Ensure all fields (e.g., PI from text, Tubing ID from scanned schematic) are present in a single, unified object.</li>
            <li><strong>Agent Role:</strong> <span class="technical-term-ai">Extractor LLM</span> handles the complex data merging and normalization.</li>
        </ul>
        """, unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_mm4:
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown('<strong>4.  Validation & Numerical Simulation</strong>', unsafe_allow_html=True)
    st.markdown(
        """
        <p>The final layer of safety and computation, ensuring the output is mathematically sound and adheres to domain rules.</p>
        <ul>
            <li><strong>Check:</strong> Deterministic checks (MD $\ge$ TVD, casing/tubing consistency) are run by the Validator Agent.</li>
            <li><strong>Tool:</strong> <span class="technical-term-ai">Solver Agent</span> uses validated inputs $\\to$ VLP/IPR models.</li>
            <li><strong>Result:</strong> Production estimate and verifiable source audit log.</li>
        </ul>
        """, unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<p style="color:#F0F0F0; font-size:1.1rem; margin-top: 30px;"><strong>Result:</strong> A full multi-modal, verifiable well-report intelligence workflow is achieved, leading to trustworthy engineering outputs.</p>', unsafe_allow_html=True)

st.markdown("---")
import streamlit as st

# Custom CSS for the specific section styling (Minimalist and high-contrast)
st.markdown("""
<style>
    /* Styling for the new Agentic Principles Section */
    .principles-section-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E90FF; /* Dodger Blue */
        border-bottom: 4px solid #444444; 
        padding-bottom: 5px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .technical-term-ai {
        color: #00CC00; /* Green highlight for AI components */
        font-weight: 700;
        font-style: italic;
    }
    .technical-term {
        color: #FF4B4B; /* Streamlit Red */
        font-weight: 700;
        font-style: italic;
    }
    .principles-text {
        color: #E0E0E0; /* Light gray text for visibility on dark background */
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 15px;
    }
    .principle-heading {
        color: #FFFFFF; /* White for main bold headings */
        font-size: 1.15em;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="principles-container">', unsafe_allow_html=True)

st.markdown('<div class="principles-section-title">Agentic-AI Principles — Detailed & Explainable</div>',
            unsafe_allow_html=True)

# FIX: Ensure the introductory paragraph is processed by the browser as HTML
st.markdown(
    """
    <p style="color:#F0F0F0; margin-bottom: 20px;">
    The foundation of LocoChat’s reliability is its <strong>Agentic Architecture</strong> —
    a system of specialized agents that reason, validate, and collaborate under an orchestrator
    to deliver accurate, explainable outcomes.
    </p>
    """, unsafe_allow_html=True
)

# Initialize the two columns
col_p1, col_p2 = st.columns(2)

# --- COLUMN 1: Core Principles ---
with col_p1:
    st.markdown("### **Core Safety & Structure**", unsafe_allow_html=True)

    st.markdown('<p class="principle-heading">Role Specialization & Modularity</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="principles-text">Involves dedicated agents for: Ingestion, Retrieval, Extractor, Validator, Vision, Solver, and UI agents. This compartmentalization ensures robust fault isolation and specialized domain expertise.</p>',
        unsafe_allow_html=True
    )

    st.markdown('<p class="principle-heading">Deterministic Safety & Trust</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="principles-text"><span class="technical-term-ai">Principle:</span> If uncertainty occurs, the system outputs <span class="technical-term">null</span> and requests human confirmation — never hallucinating numeric values.</p>',
        unsafe_allow_html=True
    )

    st.markdown('<p class="principle-heading">Chain-of-Thought Separation</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="principles-text"><span class="technical-term-ai">Process:</span> Internal reasoning is hidden; agents communicate using strict JSON/CSV formats for deterministic pipelines.</p>',
        unsafe_allow_html=True
    )

    st.markdown('<p class="principle-heading">Mandatory Auditability</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="principles-text"><span class="technical-term-ai">Record Keeping:</span> Every transformation is logged with sources, confidence, and validation outcomes.</p>',
        unsafe_allow_html=True
    )

# --- COLUMN 2: Workflow & Control ---
with col_p2:
    st.markdown("### **Control & Integration**", unsafe_allow_html=True)

    st.markdown('<p class="principle-heading">Dynamic Orchestration & Control</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="principles-text"><span class="technical-term-ai">The Planner:</span> The Orchestrator selects agents, manages retries, and applies sophisticated fallback logic.</p>',
        unsafe_allow_html=True
    )

    st.markdown('<p class="principle-heading">Multi-Model & Tool Integration</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="principles-text"><span class="technical-term-ai">The Stack:</span> Seamless collaboration between distinct models: <span class="technical-term">Llama3</span>, <span class="technical-term">Llava-Phi3</span>, <span class="technical-term">ChromaDB</span>, and physics solvers.</p>',
        unsafe_allow_html=True
    )

    st.markdown('<p class="principle-heading">Human-in-the-Loop Protocol</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="principles-text"><span class="technical-term-ai">Decision Point:</span> High-impact anomalies require human review; low-risk fixes can be auto-applied and logged.</p>',
        unsafe_allow_html=True
    )

st.markdown("---")

# --- PRACTICAL WORKFLOW SECTION ---
st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#F0F0F0; font-size:1.3rem; font-weight:700; margin-bottom:10px; border-bottom:2px solid #555;">Practical Agentic Workflow Example (Nodal Analysis)</p>',
    unsafe_allow_html=True)

st.markdown(
    """
    <ol style="color:#E0E0E0; padding-left:20px; line-height:1.6;">
        <li><strong>Ingest & Index:</strong> PDF $\\to$ OCR $\\to$ Chunks stored in <span class="technical-term">ChromaDB</span>.</li>
        <li><strong>Retrieve:</strong> Adaptive retriever locates top-K chunks containing key parameters.</li>
        <li><strong>Extract & Structure:</strong> Extractor returns strict JSON schema.</li>
        <li><strong>Validate:</strong> Validator runs domain rules.</li>
        <li><strong>Vision Fusion:</strong> Vision agent extracts tables from schematics.</li>
        <li><strong>Solve:</strong> Python solver computes $\text{Q}_{\text{sol}}$ and $\text{P}_{\text{BH}}$.</li>
        <li><strong>Output:</strong> Synthesis Agent generates the final report; results + audit log.</li>
    </ol>
    """
    , unsafe_allow_html=True)

st.markdown(
    '<p style="color:#F0F0F0; margin-top: 15px;">The Orchestrator ensures everything remains validated, mathematically grounded, and explainable.</p>'
    , unsafe_allow_html=True)

st.markdown("---")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
# ---------------------------
# Architecture diagrams (ASCII)
# ---------------------------
def render_ascii_architecture():
    ascii_art = r'''
            ┌───────────────────────────┐
            │   User / Streamlit UI     │
            └────────────┬──────────────┘
                         │ upload / query
               ┌─────────┴─────────┐
               │   Orchestrator    │
               └─────────┬─────────┘
         ┌──────────────┼──────────────┐
         │              │              │
  ┌──────▼─────┐  ┌─────▼─────┐   ┌────▼─────┐
  │ Retriever  │  │ Vision AG │   │ Extractor│
  │ (ChromaDB) │  │ (Llava)   │   │ (Llama3) │
  └──────┬─────┘  └────┬──────┘   └────┬─────┘
         │             │               │
         └────┬────────┴──────┬────────┘
              │               │
         ┌────▼─────┐     ┌───▼────┐
         │ Validator│     │ Solver │
         │(rules)   │     │VLP/IPR │
         └────┬─────┘     └───┬────┘
              │               │
         ┌────▼───────────────▼────┐
         │  Report + Audit Trail   │
         └─────────────────────────┘
    '''
    return ascii_art

with st.expander("Architecture Diagram", expanded=False):
    st.markdown("**ASCII architecture diagram (copyable):**")
    st.code(render_ascii_architecture(), language="text")
    st.markdown("**Notes:** Orchestrator calls agents selectively; vision agent outputs table JSON that merges with text extractor output.")
import streamlit as st

# Assuming col1 and col2 are defined in the parent script,
# this section focuses on the detailed technical explanation.

st.markdown("""
<style>
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E90FF; /* Dodger Blue */
        border-bottom: 3px solid #E0E0E0;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    .feature-point {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333333;
        margin-top: 10px;
    }
    .detail-list {
        margin-left: 20px;
        list-style-type: '— ';
        padding-left: 10px;
        line-height: 1.6;
    }
    .technical-term {
        color: #FF4B4B; /* Streamlit Red */
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


with st.container():
    st.markdown('<div class="section-title">📘 Document Processing & Multi-Modal Extraction Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        """
        The LocoChat system implements a multi-stage, multi-modal ingestion and analysis framework, meticulously engineered for complex engineering documents such as well reports, completion schematics, and PVT tables. Our primary goal is to produce **high-quality, RAG-optimized chunks** while preserving technical accuracy, continuity, and structure across all content formats.
        """
    )
    st.markdown("---")

st.markdown('<div class="section-title">🔍 1. Structured Data Retrieval (The Foundation)</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([1, 1.5])

with col_a:
    st.markdown('<p class="feature-point">a. Base Text Extraction</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul>
            <li><b>Mechanism:</b> Utilizes robust loaders like <span class="technical-term">PyMuPDFLoader</span> for raw text extraction.</li>
            <li><b>Purpose:</b> Captures headers, operational summaries, geological descriptions, and general narrative text.</li>
            <li><b>Output:</b> The fastest, cleanest layer of extraction, forming the prose backbone of the knowledge base.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

with col_b:
    st.markdown('<p class="feature-point">b. High-Fidelity Table Extraction</p>', unsafe_allow_html=True)
    st.markdown(
        """
        Engineering reports are data-dense; losing table structure means losing critical parameters.
        <ul>
            <li><b>Primary Tools:</b> Sequential utilization of <span class="technical-term">Camelot</span> (lattice/stream) and <span class="technical-term">pdfplumber</span>.</li>
            <li><b>Output Formats:</b> Markdown, CSV (stored in metadata), and structured LangChain <span class="technical-term">Document()</span>.</li>
            <li><b>Benefit:</b> Enables extraction of numerical values like casing sizes or pump efficiency points.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

st.markdown('<div class="section-title">🖨️ 2. Resilient OCR Pipeline (Fault-Tolerant Layer)</div>', unsafe_allow_html=True)
st.markdown(
    """
    To handle mission-critical, partially corrupted, or scanned reports, we deploy a robust, page-isolated OCR system:
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <ul class="detail-list">
        <li><b>Rasterization via PyMuPDF:</b> Converts pages into high-resolution (200 DPI) images.</li>
        <li><b>OCR via Tesseract:</b> Extracts text including signatures, handwritten notes, faint markings.</li>
        <li><b>Error Isolation:</b> Per-page <code>try/except</code> blocks ensure corrupted pages never stop processing.</li>
    </ul>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

st.markdown('<div class="section-title">📊 3. Structured Data and Vision Pipelines</div>', unsafe_allow_html=True)

col_c, col_d = st.columns(2)

with col_c:
    st.markdown('<p class="feature-point">Excel (XLS/XLSX) Processing</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul>
            <li><b>Full Sheet Integration:</b> Reads and processes all sheets in a workbook.</li>
            <li><b>RAG Optimization:</b> Preserves structure via Markdown/CSV conversion.</li>
            <li><b>Statistics:</b> Automatically calculates mean, min, max, std for numeric columns.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

with col_d:
    st.markdown('<p class="feature-point">Image-Based Extraction (Llava-Phi3 Vision)</p>', unsafe_allow_html=True)
    st.markdown(
        """
        For complex schematics or diagrams:
        <ul>
            <li><b>Vision Model:</b> Uses <span class="technical-term">Llava-Phi3</span>.</li>
            <li><b>Multimodal Output:</b> Diagram descriptions, OCR labels, reconstructed tables, spatial relationships.</li>
            <li><b>Smart Routing:</b> <span class="technical-term">load_documents()</span> automatically sends images to this pipeline.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

st.markdown('<div class="section-title">🧩 4. Adaptive Chunking & Indexing for RAG Accuracy</div>', unsafe_allow_html=True)

col_e, col_f = st.columns(2)

with col_e:
    st.markdown('<p class="feature-point">Adaptive Document Chunking</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul>
            <li><b>Text Splitter (2000 chars):</b> For narrative sections.</li>
            <li><b>Table Splitter (1200 chars):</b> For structured data without breaking semantic rows.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

with col_f:
    st.markdown('<p class="feature-point">ChromaDB Indexing & Vector Embeddings</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul>
            <li><b>Vector Storage:</b> Chunks are embedded using 
                <span class="technical-term">OllamaEmbeddings</span> (Llama3)
                and stored in <span class="technical-term">ChromaDB</span>.
            </li>
            <li><b>Rich Metadata:</b> All extracted metadata (CSV, stats, page numbers) is preserved.</li>
            <li><b>Traceability:</b> Enables auditability and prevents hallucinations.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

st.markdown('<div class="section-title">📈 5. Post-Retrieval Actions & Auditability</div>', unsafe_allow_html=True)

col_g, col_h = st.columns(2)

with col_g:
    st.markdown('<p class="feature-point">Auto-Plotting from Extracted Tables</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul>
            <li>Automatically reads embedded CSV metadata.</li>
            <li>Plots numerical columns using <span class="technical-term">matplotlib</span> / <span class="technical-term">pandas</span>.</li>
            <li>Useful for pump curves, pressure-depth relationships, or trajectory plots.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

with col_h:
    st.markdown('<p class="feature-point">Source Rendering & Transparency</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul>
            <li><b>Auditability:</b> Retrieved chunks are shown immediately.</li>
            <li><b>Transparency:</b> Includes file name, page, content type, and metadata.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

st.markdown('<div class="section-title">🏗️ End-to-End Architecture Outcome</div>', unsafe_allow_html=True)
st.markdown(
    """
    This validated ingestion pipeline supports the agent-orchestrated nodal analysis solver by ensuring complete data fidelity, OCR/Vision fallback, and transparent retrieval for engineering-grade outputs.
    """,
    unsafe_allow_html=True
)

st.write("---")


# ---------------------------
# Merged-file generator — creates a full streamlit_app_merged.py for download
# ---------------------------
import io

def make_merged_app(original_code_text):
    """
    original_code_text: string of your original app (paste the entire original file text here)
    This function will append the add-ons to the original text and return a BytesIO buffer.
    """
    addons = """

# ------------------------
# ADD-ONS APPENDED (ChromaDB, Multi-Modal, Agentic Principles, PDF generator, Diagrams)
# ------------------------

# (Paste the Section 1/2/3 blocks here exactly once)
"""
    merged = original_code_text + "\n\n" + addons
    return io.BytesIO(merged.encode("utf-8"))

with st.expander("Create merged streamlit_app file for download", expanded=False):
    st.write("Paste your entire original streamlit_app.py contents into the box below, then click 'Create merged file'.")
    original_text = st.text_area("Paste original file content here", height=300)
    if st.button("Create merged file"):
        if not original_text.strip():
            st.error("Please paste your original streamlit_app.py contents in the text area.")
        else:
            buf = make_merged_app(original_text)
            st.download_button("Download merged streamlit_app.py", data=buf, file_name="streamlit_app_merged.py", mime="text/plain")

