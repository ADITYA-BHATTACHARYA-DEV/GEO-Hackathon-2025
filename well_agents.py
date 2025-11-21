from typing import List, Dict, Any
from pydantic import BaseModel, Field, ValidationError
import numpy as np
import json

# Attempt to import the calculation module
try:
    from NodalAnalysis import perform_nodal_analysis
except ImportError:
    # Define a dummy function if NodalAnalysis.py is missing to prevent crash
    def perform_nodal_analysis(**kwargs):
        return {"status": "error", "message": "NodalAnalysis.py not found or failed to import."}


# --- Pydantic Schemas for Structured Data Exchange ---
class TrajectoryPoint(BaseModel):
    """Defines a segment point in the wellbore trajectory."""
    MD: float = Field(description="Measured Depth in meters.")
    TVD: float = Field(description="True Vertical Depth in meters.")
    ID: float = Field(description="Inner Diameter of casing/tubing in meters.")


class WellParameters(BaseModel):
    """The master schema for all Nodal Analysis inputs."""
    reservoir_pressure: float = Field(description="Reservoir pressure (bar).")
    wellhead_pressure: float = Field(description="Wellhead pressure (bar).")
    PI: float = Field(description="Productivity Index (m3/hr per bar).")
    esp_depth: float = Field(description="ESP intake depth (meters).")
    well_trajectory: List[TrajectoryPoint] = Field(description="List of casing/tubing segments.")

    # Simplified constants (retrieved by RAG/defaults)
    rho: float = 1000.0
    mu: float = 1e-3
    roughness: float = 1e-5
    pump_curve: dict = Field(default={"flow": [0, 100, 200, 300, 400], "head": [600, 550, 450, 300, 100]})


# --- Agent Functions ---

def validate_well_parameters(params: WellParameters) -> dict:
    """
    Data Validation Agent: Performs sanity checks on extracted well geometry and physics.
    """
    validation_report = {"errors": [], "warnings": [], "validated_params": params.dict()}

    # RULE 1: MD must be >= TVD (Critical Geometry Check)
    for i, point in enumerate(params.well_trajectory):
        if point.MD < point.TVD - 1.0:  # Allowing for a small tolerance
            validation_report["errors"].append(
                f"Critical Error: MD ({point.MD:.1f}m) is significantly less than TVD ({point.TVD:.1f}m) at segment {i}. Check for unit mix-up or error."
            )

    # RULE 2: Check for minimum depth (Must be deeper than ESP intake)
    if params.well_trajectory:
        final_tvd = params.well_trajectory[-1].TVD
        if final_tvd < params.esp_depth:
            validation_report["errors"].append(
                f"Critical Error: Final TVD ({final_tvd:.1f}m) is shallower than ESP depth ({params.esp_depth:.1f}m). Pump location is invalid."
            )

    # RULE 3: Check PI range
    if params.PI < 0.1 or params.PI > 50.0:
        validation_report["warnings"].append(
            f"Warning: PI ({params.PI:.1f}) is outside typical range (0.1 - 50)."
        )

    return validation_report


def run_nodal_analysis_tool(validated_params: WellParameters) -> dict:
    """
    Well Flow Model Agent: Executes the Nodal Analysis calculation using the external script.
    """
    try:
        # Pass parameters directly from the Pydantic model to the function
        results = perform_nodal_analysis(**validated_params.dict(), plot_results=False)

        return {"status": "success", "results": results}

    except Exception as e:
        return {"status": "error", "message": f"Nodal Analysis failed during execution: {str(e)}"}