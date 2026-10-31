from mequa import eval_agent, eval_blockspec


def test_agent_eval_harness_dry_run_scores_a_correct_scripted_model():
    out = eval_agent.main(["--dry-run", "--phrasings", "1"])
    s = out["summary"]["scripted-dry-run"]
    assert s["completed"] == 1.0 and s["threshold_correct"] == 1.0 and s["template_fallback_rate"] == 0.0


def test_agent_scoring_marks_a_missed_threshold():
    truth = {"threshold": 0.05}
    result = {"status": "ok", "iterations": 2, "grounding": {"attempts": [{"ungrounded": ["9.9"]}], "fallback": True},
              "summary": "nothing useful", "trace": [{"type": "tool_call", "tool": "sweep_load",
                                                       "result": {"rows": [], "threshold_load": None}}]}
    row = eval_agent.score_run(result, truth, 1.0)
    assert not row["threshold_correct"] and row["first_attempt_ungrounded"] and row["template_fallback"]


def test_blockspec_eval_dry_run_matches_analytic_answers():
    s = eval_blockspec.main(["--dry-run"])["scripted-dry-run"]
    assert s["valid"] == 1.0 and s["matches_maths"] == 1.0 and s["coupled"] == 1.0


def test_blockspec_eval_catches_a_wrong_spec():
    case = eval_blockspec.build_cases()[0]  # first order, K=2, tau=1.5
    bad = {"blocks": [{"id": "s", "type": "step_input", "parameters": {"final_value": 1.0}},
                      {"id": "tf", "type": "transfer_function", "parameters": {"numerator": [9.0], "denominator": [1.5, 1.0]}}],
           "connections": [{"from": "s", "to": "tf"}]}
    from mequa import backend as backend_mod, numpy_backend as nb
    res = eval_blockspec.check_case(case, bad, nb.synthetic_cells(100, seed=1, clusters=1), backend_mod.load("numpy"))
    assert res["stages"]["valid"] and res["stages"]["simulated"] and not res["stages"]["matches_maths"]


def test_a_flat_signal_with_the_right_final_value_no_longer_passes_the_maths_check():
    case = eval_blockspec.build_cases()[2]  # first order, K=3, tau=2: y(10) = 2.98
    flat = {"blocks": [{"id": "s", "type": "step_input", "parameters": {"step_time": 0.0, "initial_value": 2.98, "final_value": 2.98}},
                       {"id": "tf", "type": "transfer_function", "parameters": {"numerator": [1.0], "denominator": [1e-3, 1.0]}}],
            "connections": [{"from": "s", "to": "tf"}]}
    from mequa import backend as backend_mod, numpy_backend as nb
    res = eval_blockspec.check_case(case, flat, nb.synthetic_cells(100, seed=1, clusters=1), backend_mod.load("numpy"))
    assert res["stages"]["simulated"] and not res["stages"]["matches_maths"]
