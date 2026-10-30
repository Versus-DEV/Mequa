"""Tool schemas the Llama investigator may call. Both tools run the verified spring-network solver."""

from __future__ import annotations

from typing import Any, Dict, List


def get_solver_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "run_solver",
                "description": (
                    "Run one relaxation of the cell spring network at a given load and return strain, SPHI and "
                    "per-zone statistics. load_fraction is the fraction of the tissue width the right edge is pulled."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "load_fraction": {"type": "number", "description": "0.005 to 0.2"},
                        "zones": {"type": "integer", "description": "density zones, 2 to 8 (optional)"},
                        "neighbors": {"type": "integer", "description": "neighbors per cell, 3 to 14 (optional)"},
                    },
                    "required": ["load_fraction"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sweep_load",
                "description": (
                    "Solve many loads between start and stop and report where the 95th percentile cell strain "
                    "reaches the strain limit (threshold_load). Use it to find the failure threshold."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "number", "description": "first load fraction, at least 0.005"},
                        "stop": {"type": "number", "description": "last load fraction, at most 0.2"},
                        "steps": {"type": "integer", "description": "number of loads, 3 to 16"},
                    },
                    "required": ["start", "stop", "steps"],
                },
            },
        },
    ]
