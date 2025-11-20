
# agent_loop.py

import matplotlib.pyplot as plt
import io
import base64

from agent_components import get_parameters_from_rag, run_nodal_analysis

def plan_for_nodal():
    return [
        "Identify required parameters: rho, mu, roughness, reservoir_pressure, wellhead_pressure, PI, esp_depth, pump_curve, trajectory.",
        "Query and retrieve candidate documents via RAG.",
        "Extract parameters from metadata, tables (CSV), and text. Apply validation and defaults.",
        "Run nodal analysis to compute VLP/IPR and operating point.",
        "Format an accurate, user-understandable summary and include a plot."
    ]

def execute_nodal_rag(source_docs, overrides=None):
    # Step 1–3: retrieve/extract/validate
    params = get_parameters_from_rag(source_docs)
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})

    # Step 4: compute
    results = run_nodal_analysis(params)

    # Step 5: craft response
    summary_lines = []
    summary_lines.append("Nodal analysis summary")
    summary_lines.append(f"- Density (rho): {params['rho']} kg/m3")
    summary_lines.append(f"- Viscosity (mu): {params['mu']} Pa.s")
    summary_lines.append(f"- Roughness: {params['roughness']} m")
    summary_lines.append(f"- Reservoir pressure: {params['reservoir_pressure']} bar")
    summary_lines.append(f"- Wellhead pressure: {params['wellhead_pressure']} bar")
    summary_lines.append(f"- Productivity Index (PI): {params['PI']} m3/hr/bar")
    summary_lines.append(f"- ESP depth: {params['esp_depth']} m")
    summary_lines.append(f"- Pump curve points: {len(params['pump_curve']['flow'])}")
    summary_lines.append(f"- Trajectory segments: {max(len(params['trajectory'])-1, 0)}")

    if results["sol_flow"] is not None:
        summary_lines.append("")
        summary_lines.append("Operating point:")
        summary_lines.append(f"- Flowrate Q: {results['sol_flow']:.2f} m3/hr")
        summary_lines.append(f"- Bottomhole pressure (BHP): {results['sol_pbh']:.2f} bar")
        if results["sol_head"] is not None:
            summary_lines.append(f"- Pump head: {results['sol_head']:.1f} m")
    else:
        summary_lines.append("")
        summary_lines.append("No operating point found within tolerance. Consider adjusting PI, pump curve, or trajectory.")

    summary_text = "\n".join(summary_lines)

    # Build plot and return as base64 to embed or save
    fig, ax = plt.subplots()
    ax.plot(results["flows"], results["p_vlp"], label="VLP")
    ax.plot(results["flows"], results["p_ipr"], label="IPR")
    if results["sol_flow"] is not None:
        ax.scatter(results["sol_flow"], results["sol_pbh"], color="red", label="Operating point")
    ax.set_xlabel("Flowrate [m3/hr]")
    ax.set_ylabel("Pressure [bar]")
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plot_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return summary_text, plot_b64, params, results
