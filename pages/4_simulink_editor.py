"""Simulink/Xcos Editor page: prompt-to-model translation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from mequa.blockspec.semantic_block_translator import translate_to_block_spec
from mequa.blockspec.block_validator import validate_block_spec
from mequa.blockspec.xcos_importer import simulate_block_spec
from mequa.utils.env import load_env_file


from mequa import ui  # noqa: E402

load_env_file()

ui.page_header('Simulink / Xcos Editor', 'Prompt-to-model translation with a validation loop.')

with st.sidebar:
    st.header("Model Description")

    user_prompt = st.text_area(
        "Describe the system",
        value="Simulate a first-order transfer function with gain 2 and time constant 1, driven by a step input.",
        height=150,
    )

    generate = st.button("Generate Block Spec", type="primary")


if generate:
    with st.spinner("Translating to block specification..."):
        translation = translate_to_block_spec(user_prompt)

    st.session_state["translation"] = translation


translation = st.session_state.get("translation")

if translation is None:
    st.info("Describe a system and click Generate block spec.")
    st.stop()


tab_spec, tab_validation, tab_simulation = st.tabs(
    ["Block Spec", "Validation", "Simulation"]
)

with tab_spec:
    if translation.get("success"):
        st.json(translation["block_spec"])

        with st.expander("Raw Llama response"):
            st.code(translation.get("raw", ""))
    else:
        st.error(translation.get("error", "Unknown error"))

        if "raw" in translation:
            with st.expander("Raw Llama response"):
                st.code(translation["raw"])

with tab_validation:
    if translation.get("success"):
        validation = validate_block_spec(translation["block_spec"])

        if validation["valid"]:
            st.success("Block specification is valid.")
        else:
            st.error("Validation errors:")
            for error in validation["errors"]:
                st.write(f"- {error}")

        st.json(validation)
    else:
        st.warning("No valid block spec to validate.")

with tab_simulation:
    if translation.get("success"):
        validation = validate_block_spec(translation["block_spec"])

        if validation["valid"]:
            with st.spinner("Simulating..."):
                simulation = simulate_block_spec(translation["block_spec"])

            if simulation.get("success"):
                st.success("Simulation successful.")

                # Plot results.
                import numpy as np

                time_data = np.array(simulation["time"])
                output_data = np.array(simulation["output"])

                st.line_chart(
                    {"Time": time_data, "Output": output_data},
                    x="Time",
                    y="Output",
                )

                st.json(
                    {
                        "block_id": simulation.get("block_id"),
                        "num_points": len(time_data),
                        "final_value": float(output_data[-1]) if len(output_data) > 0 else None,
                    }
                )
            else:
                st.error(simulation.get("error", "Unknown simulation error"))
        else:
            st.warning("Fix validation errors before simulating.")
    else:
        st.warning("No valid block spec to simulate.")