"""Build results/RESULTS.md from the result files, so every number in the README comes from a file.

Anything not measured yet is printed as "not measured", never filled in. The ablation table
answers "what breaks or degrades if this Meta model is removed" and only states what a result
file supports.

Run:  python -m mequa.make_report
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path("results")


def _load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _pct(x):
    return "not measured" if x is None else f"{round(100 * x)}%"


def firewall_section(lines):
    data = _load("eval_all.json")
    lines += ["## Firewall (Llama Guard + Llama intent classifier + rules)", ""]
    if not data:
        lines += ["not measured", ""]
        return None
    lines += ["| Set | Policy | Clinical blocked | Research wrongly blocked |", "|---|---|---|---|"]
    for set_name, pol in data.items():
        for name, s in pol.items():
            n_c = s["clinical_blocked"] + s["clinical_missed"]
            n_r = s["research_wrongly_blocked"] + s["research_passed"]
            lines.append(f"| {set_name} | {name} | {s['clinical_blocked']}/{n_c} | {s['research_wrongly_blocked']}/{n_r} |")
    lines += ["", "Sets are small and written by the author; a perfect score on n items still allows a true miss rate of about 3/n.", ""]
    return data


def agent_section(lines):
    data = _load("agent_eval.json")
    lines += ["## Llama tool-calling investigator", ""]
    if not data:
        lines += ["not measured", ""]
        return None
    cfg = data["config"]
    lines += [f"Ground truth: fine serial sweep, {cfg['cells']} cells, strain limit {cfg['strain_limit']}, seeds {cfg['seeds']}. "
              "Temperature 0, so variety comes from seeds and phrasings.", "",
              "| Model | Runs | Completed | Called sweep | Threshold correct | Invented number on 1st try | Template fallback | Median s |",
              "|---|---|---|---|---|---|---|---|"]
    for m, s in data["summary"].items():
        lines.append(f"| {m} | {s['runs']} | {_pct(s['completed'])} | {_pct(s['called_sweep'])} | {_pct(s['threshold_correct'])} | "
                     f"{_pct(s['first_attempt_invented_number_rate'])} | {_pct(s['template_fallback_rate'])} | {s['median_seconds']} |")
    lines.append("")
    return data


def blockspec_section(lines):
    data = _load("blockspec_eval.json")
    lines += ["## Llama-written block specifications", ""]
    if not data:
        lines += ["not measured", ""]
        return None
    lines += ["| Model | Prompts | JSON | Valid | Simulates | Matches maths | Drives strain solver |", "|---|---|---|---|---|---|---|"]
    for m, s in data["summary"].items():
        lines.append(f"| {m} | {s['prompts']} | {_pct(s['json'])} | {_pct(s['valid'])} | {_pct(s['simulated'])} | {_pct(s['matches_maths'])} | {_pct(s['coupled'])} |")
    lines.append("")
    return data


def segmentation_section(lines):
    lines += ["## Segmentation (SAM vs classical, F1 against expert masks)", ""]
    sam = _load("seg_eval_sam.json")
    if not sam:
        lines += ["not measured", ""]
        return None
    lines += ["```", json.dumps(sam, indent=2)[:3000], "```", ""]
    return sam


def ablation(lines, fw, agent, spec, seg):
    lines += ["## Ablation: what changes without each Meta model", "",
              "| Meta model | Job | Without it | Evidence |", "|---|---|---|---|"]
    if fw:
        last = [k for k in fw if k.startswith("round4")] or [k for k in fw if k.startswith("round3")]
        key = last[-1] if last else list(fw)[-1]
        p = fw[key]
        def miss(name):
            return f"{p[name]['clinical_missed']} clinical requests missed"
        intent_row = f"Llama 3.2 intent classifier | blocks clinical-advice requests | without the intent layer the other layers on their own miss: {miss('guard_s6_only')} (Guard alone), {miss('rules_only')} (rules alone) | results/eval_all.json, {key}"
        guard_row = f"Llama Guard 3 | S6 category screen on requests and replies | Guard alone: {miss('guard_s6_only')}; by design the reply check also uses Guard, so replies would go unscreened | results/eval_all.json, {key}"
    else:
        intent_row = "Llama 3.2 intent classifier | blocks clinical-advice requests | not measured | -"
        guard_row = "Llama Guard 3 | S6 category screen on requests and replies | not measured | -"
    lines += [f"| {intent_row} |", f"| {guard_row} |"]
    if agent:
        parts = "; ".join(f"{m}: threshold correct {_pct(s['threshold_correct'])}, invented number first try {_pct(s['first_attempt_invented_number_rate'])}"
                          for m, s in agent["summary"].items())
        lines.append(f"| Llama (3B / 8B) tool-calling | plans solver calls, finds the failure threshold | no planner means no autonomous threshold search; with it: {parts} | results/agent_eval.json |")
    else:
        lines.append("| Llama (3B / 8B) tool-calling | plans solver calls, finds the failure threshold | not measured | - |")
    if spec:
        parts = "; ".join(f"{m}: {_pct(s['matches_maths'])} match the analytic answer" for m, s in spec["summary"].items())
        lines.append(f"| Llama block-spec writer | plain English to Simulink-style spec | without it the user writes the spec by hand; with it: {parts} | results/blockspec_eval.json |")
    else:
        lines.append("| Llama block-spec writer | plain English to Simulink-style spec | not measured | - |")
    if seg:
        lines.append("| SAM | image to cell masks | classical thresholding instead; F1 difference in the segmentation section | results/seg_eval_sam.json |")
    else:
        lines.append("| SAM | image to cell masks | not measured | - |")
    lines.append("")


def main():
    lines = ["# Mequa measured results", "", "Generated by `python -m mequa.make_report` from files in results/. Nothing here is typed by hand.", ""]
    fw = firewall_section(lines)
    agent = agent_section(lines)
    spec = blockspec_section(lines)
    seg = segmentation_section(lines)
    ablation(lines, fw, agent, spec, seg)
    R.mkdir(exist_ok=True)
    (R / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()