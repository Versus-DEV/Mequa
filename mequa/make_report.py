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
            lines.append(f"| {set_name.split(' ')[0]} | {name} | {s['clinical_blocked']}/{n_c} | {s['research_wrongly_blocked']}/{n_r} |")
    lines += ["", "Sets are small and written by the author; a perfect score on n items still allows a true miss rate of about 3/n.", ""]
    return data


def _agent_table(lines, data):
    lines += ["| Model | Runs | Completed | Model swept | Forced | Threshold correct | Correct, model-initiated | Invented number on 1st try | Template fallback | Median s |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for m, s_ in data["summary"].items():
        lines.append(f"| {m} | {s_['runs']} | {_pct(s_['completed'])} | {_pct(s_['called_sweep'])} | {_pct(s_.get('forced_sweep_rate'))} | {_pct(s_['threshold_correct'])} | "
                     f"{_pct(s_.get('threshold_correct_model_initiated'))} | {_pct(s_['first_attempt_invented_number_rate'])} | {_pct(s_['template_fallback_rate'])} | {s_['median_seconds']} |")
    lines.append("")


def agent_section(lines):
    data = _load("agent_eval.json")
    lines += ["## Llama tool-calling investigator", ""]
    if not data:
        lines += ["not measured", ""]
        return None
    cfg = data["config"]
    meshes = sorted({r.get("mesh", "synthetic") for r in data.get("rows", [])})
    lines += [f"Ground truth: fine serial sweep (40 loads), strain limit {cfg['strain_limit']} on synthetic meshes of {cfg['cells']} cells; "
              "temperature 0, so variety comes from meshes and phrasings. 'Model swept' counts runs where the model itself called sweep_load; "
              "'forced' counts runs where the program had to run a default sweep after the model ignored two requests.", ""]
    _agent_table(lines, data)
    real = _load("agent_eval_real.json")
    if real:
        lines += ["On real detected cells (the strain limit is chosen automatically so the strain crosses it):", ""]
        _agent_table(lines, real)
    v1 = _load("agent_eval_v1.json")
    if v1:
        lines += ["First measurement, before the sweep-nudge fix (kept for the record):", "",
                  "| Model | Runs | Called sweep | Threshold correct | Template fallback |", "|---|---|---|---|---|"]
        for m, x in v1["summary"].items():
            lines.append(f"| {m} | {x['runs']} | {_pct(x['called_sweep'])} | {_pct(x['threshold_correct'])} | {_pct(x['template_fallback_rate'])} |")
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
    v1 = _load("blockspec_eval_v1.json")
    if v1:
        lines += ["First measurement, before standard forms and the repair retry (kept for the record):", "",
                  "| Model | Prompts | Valid | Matches maths |", "|---|---|---|---|"]
        for m, x in v1["summary"].items():
            lines.append(f"| {m} | {x['prompts']} | {_pct(x['valid'])} | {_pct(x['matches_maths'])} |")
        lines.append("")
    return data


def segmentation_section(lines):
    lines += ["## Segmentation (F1 against expert masks, BBBC039)", ""]
    sam = _load("seg_eval_sam.json")
    cls = _load("seg_eval_classical.json")
    if not sam and not cls:
        lines += ["not measured", ""]
        return None
    rows = []
    for d in (cls, sam):
        if d:
            rows += [r for r in d["results"] if r.get("method") != "ground_truth"]
    gt = next((r for d in (sam, cls) if d for r in d["results"] if r.get("method") == "ground_truth"), None)
    if gt:
        lines += [f"Image {gt.get('image', '?')}: {gt['instances']} expert-annotated nuclei. Match radius {(sam or cls)['radius_px']} px.", ""]
    lines += ["| Method | Detected | Precision | Recall | F1 | Count error | Seconds |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['detected']} | {r['precision']} | {r['recall']} | {r['f1']} | {r['count_error_pct']}% | {r.get('seconds', '-')} |")
    lines.append("")
    return {"sam": sam, "classical": cls}


def mpi_section(lines):
    data = _load("mpi_scaling.json")
    lines += ["## MPI parameter-sweep scaling", ""]
    if not data:
        lines += ["not measured (run the clean measurement in docs/RUN_THESE.md)", ""]
        return None
    lines += ["```", json.dumps(data, indent=2)[:2500], "```", "",
              "This is parameter-sweep parallelism: each rank solves different loads. See docs/DOMAIN_DECOMPOSITION.md for the domain-decomposed variant.", ""]
    ds = _load("domain_scaling.json")
    if ds:
        lines += [f"Domain-decomposed single solve ({ds['cells']} cells, {ds['dim']}D, {ds['iters']} iterations, {ds['repeats']} interleaved repeats; serial median {ds['serial_seconds_median']} s):", "",
                  "| Ranks | Wall s (median) | Speedup vs serial | Efficiency | Max position diff vs serial | Noisy |", "|---|---|---|---|---|---|"]
        for r in ds["scaling"]:
            lines.append(f"| {r['ranks']} | {r['wall_seconds_median']} | {r['speedup_vs_serial']}x | {r['efficiency_vs_serial']} | {r['max_abs_position_diff_vs_serial']} | {r['noisy']} |")
        lines.append("")
    dom = _load("domain_mpi.json")
    if dom and not ds:
        lines += ["Domain-decomposed single solve (halo exchange), compared with the serial solve:", "", "```", json.dumps(dom, indent=2), "```", ""]
    return data


def real3d_section(lines):
    data = _load("real3d.json")
    lines += ["## Real 3D nuclei (BBBC033)", ""]
    if not data:
        lines += ["not measured", ""]
        return None
    det = data.get("detection_vs_expert_labels", {})
    an = data["analysis_on_detected_cells"]
    lines += [f"Volume (z, y, x): {data['volume_shape_zyx']}. Detection against the expert 3D labels: "
              f"{det.get('detected')} detected, {det.get('ground_truth')} expert, precision {det.get('precision')}, recall {det.get('recall')}, F1 {det.get('f1')} "
              f"(match radius {det.get('radius_vox')} voxels).", "",
              f"3D strain analysis on the detected cells: {an.get('cells')} cells, {an.get('n_components')} connected component(s), load path {an.get('load_path')}, "
              f"residual {an.get('residual')}, SPHI {an.get('sphi')}, failure-threshold load {an.get('threshold_load_p95')}.", ""]
    return data


def _fw_key(fw, prefix):
    keys = [k for k in fw if k.startswith(prefix)]
    return keys[-1] if keys else None


def ablation(lines, fw, agent, spec, seg):
    lines += ["## Ablation: what changes without each Meta model", ""]
    if fw:
        lines += ["Firewall, leave-one-layer-out on the English adversarial set (round 5) and the untouched set (round 3). "
                  "Cells are clinical requests blocked / research requests wrongly blocked.", "",
                  "| Layers active | Round 3 | Round 5 |", "|---|---|---|"]
        r3, r5 = _fw_key(fw, "round3"), _fw_key(fw, "round5")
        for pol in ("combined_all_layers", "without_intent (rules+guard)", "without_guard (rules+intent)", "without_rules (guard+intent)"):
            cells = []
            for k in (r3, r5):
                if k and pol in fw[k]:
                    x = fw[k][pol]
                    cells.append(f"{x['clinical_blocked']}/{x['clinical_blocked'] + x['clinical_missed']} blocked, {x['research_wrongly_blocked']} wrongly blocked")
                else:
                    cells.append("not measured")
            lines.append(f"| {pol} | {cells[0]} | {cells[1]} |")
        lines.append("")
    lines += ["| Meta model | Job | Without it | Evidence |", "|---|---|---|---|"]
    if not fw:
        lines.append("| Llama 3.2 and Llama Guard 3 | firewall | not measured | - |")
    else:
        lines.append("| Llama 3.2 intent classifier | blocks clinical-advice requests | see the without_intent row above | results/eval_all.json |")
        lines.append("| Llama Guard 3 | S6 screen on requests and replies | see the without_guard row above; replies would also go unscreened by design | results/eval_all.json |")
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
    if seg and seg.get("sam") and seg.get("classical"):
        lines.append("| SAM | image to cell masks | classical thresholding instead; compare F1 rows in the segmentation table | results/seg_eval_*.json |")
    elif seg and seg.get("sam"):
        lines.append("| SAM | image to cell masks | classical baseline not run yet (see docs/RUN_THESE.md) | results/seg_eval_sam.json |")
    else:
        lines.append("| SAM | image to cell masks | not measured | - |")
    lines.append("")


START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def inject_readme(text):
    readme = Path("README.md")
    if not readme.exists():
        return
    body = readme.read_text(encoding="utf-8")
    if START in body and END in body:
        head, rest = body.split(START, 1)
        _, tail = rest.split(END, 1)
        shown = "\n".join(l for l in text.split("\n") if not l.startswith("# "))
        readme.write_text(head + START + "\n" + shown + "\n" + END + tail, encoding="utf-8")


def main():
    lines = ["# Mequa measured results", "", "Generated by `python -m mequa.make_report` from files in results/. Nothing here is typed by hand.", ""]
    fw = firewall_section(lines)
    agent = agent_section(lines)
    spec = blockspec_section(lines)
    seg = segmentation_section(lines)
    mpi = mpi_section(lines)
    real3d_section(lines)
    ablation(lines, fw, agent, spec, seg)
    R.mkdir(exist_ok=True)
    text = "\n".join(lines)
    (R / "RESULTS.md").write_text(text, encoding="utf-8")
    inject_readme(text)
    print(text)


if __name__ == "__main__":
    main()
