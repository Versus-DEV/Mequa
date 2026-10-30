"""When the firewall blocks a request, say so plainly and offer generic research questions.

This module never answers, rephrases or "converts" the blocked request. It does not use a model
to rewrite it. Blocking is decided by the measured firewall (rules, Llama Guard on category S6,
Llama intent classifier) and the suggestions are fixed text about datasets and simulations, so
a clinical question about a person can never be turned into an analysis of that person.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

BLOCK_MESSAGE = (
    "Mequa is a research tool for datasets, cohorts and simulations. It does not give diagnosis, "
    "prognosis or treatment guidance for any individual, so this request was blocked."
)

SUGGESTIONS: List[str] = [
    "Cluster the cells in this dataset into density zones and report the strain in each zone under a 5 percent load.",
    "Sweep the load from 1 percent to 12 percent on this dataset and report where the 95th percentile cell strain reaches 0.05.",
    "Compare how SPHI changes when the number of mechanical neighbors per cell goes from 6 to 10 on the same dataset.",
]


def pivot_to_research(prompt: str, check_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """Run the firewall. Allowed requests pass through unchanged; blocked ones get a message and
    fixed research suggestions, never a rewrite of the original request."""
    if check_fn is None:
        from mequa.firewall import check_request as check_fn
    verdict = check_fn(prompt)
    if verdict.allowed:
        return {"decision": "allowed", "degraded": bool(verdict.degraded), "flags": list(verdict.flags)}
    return {
        "decision": "blocked",
        "blocked_by": list(verdict.blocked_by),
        "message": BLOCK_MESSAGE,
        "suggestions": SUGGESTIONS,
    }
