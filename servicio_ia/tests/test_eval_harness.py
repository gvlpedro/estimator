"""Unit tests for the golden-set eval harness's pure scoring logic.

Deliberately does NOT call the real /estimate endpoint or any model --
``evaluate_case``/``build_report`` take a pre-fetched ``result`` dict, exactly
like ``call_estimate``'s real return shape, so these tests hand-build that
shape instead of hitting the network (Strict TDD Mode, no live LLM spend).
"""

from eval.run_eval import build_report, evaluate_case, load_golden_set


def _estimation_case(case_id="case-x", expected_hours=300, acceptable_range=(200, 400)):
    return {
        "id": case_id,
        "case_type": "estimation",
        "expected_hours": expected_hours,
        "acceptable_range": list(acceptable_range),
    }


def _result(status_code=200, body=None):
    return {"status_code": status_code, "body": body or {}, "latency_ms": 42.0}


# ---------------------------------------------------------------------------
# evaluate_case -- estimation cases
# ---------------------------------------------------------------------------


def test_estimation_case_passes_when_prediction_is_within_range():
    case = _estimation_case()
    result = _result(body={"status": "validated", "estimate": {"total_hours": 310}})

    evaluation = evaluate_case(case, result)

    assert evaluation["passed"] is True
    assert evaluation["predicted_hours"] == 310
    assert evaluation["abs_error"] == 10


def test_estimation_case_fails_when_prediction_is_outside_range():
    case = _estimation_case()
    result = _result(body={"status": "validated", "estimate": {"total_hours": 900}})

    evaluation = evaluate_case(case, result)

    assert evaluation["passed"] is False


def test_estimation_case_fails_when_no_estimate_is_returned():
    case = _estimation_case()
    result = _result(body={"status": "awaiting_human_review", "estimate": None})

    evaluation = evaluate_case(case, result)

    assert evaluation["passed"] is False
    assert evaluation["predicted_hours"] is None
    assert evaluation["abs_error"] is None


# ---------------------------------------------------------------------------
# evaluate_case -- abstention / partial_precedent cases
# ---------------------------------------------------------------------------


def test_abstention_case_passes_only_on_genuine_no_precedent():
    case = {"id": "case-abstain", "case_type": "abstention"}
    result = _result(
        body={
            "status": "awaiting_human_review",
            "review": {"validation": {"no_precedent": True}},
        }
    )

    evaluation = evaluate_case(case, result)

    assert evaluation["passed"] is True


def test_abstention_case_fails_when_review_has_some_precedent():
    case = {"id": "case-abstain", "case_type": "abstention"}
    result = _result(
        body={
            "status": "awaiting_human_review",
            "review": {"validation": {"no_precedent": False}},
        }
    )

    evaluation = evaluate_case(case, result)

    assert evaluation["passed"] is False


def test_abstention_case_fails_when_system_answers_confidently_instead():
    # The dangerous failure mode: the system should have paused, but didn't.
    case = {"id": "case-abstain", "case_type": "abstention"}
    result = _result(body={"status": "validated", "estimate": {"total_hours": 200}})

    evaluation = evaluate_case(case, result)

    assert evaluation["passed"] is False


def test_partial_precedent_case_passes_when_review_confirms_some_precedent():
    case = {"id": "case-edge", "case_type": "partial_precedent"}
    result = _result(
        body={
            "status": "awaiting_human_review",
            "review": {"validation": {"no_precedent": False}},
        }
    )

    evaluation = evaluate_case(case, result)

    assert evaluation["passed"] is True


def test_partial_precedent_case_fails_when_actually_no_precedent():
    case = {"id": "case-edge", "case_type": "partial_precedent"}
    result = _result(
        body={
            "status": "awaiting_human_review",
            "review": {"validation": {"no_precedent": True}},
        }
    )

    evaluation = evaluate_case(case, result)

    assert evaluation["passed"] is False


# ---------------------------------------------------------------------------
# build_report -- aggregation
# ---------------------------------------------------------------------------


def test_build_report_computes_within_range_rate_and_mean_absolute_error():
    evaluations = [
        {"type": "estimation", "passed": True, "abs_error": 10, "latency_ms": 100},
        {"type": "estimation", "passed": False, "abs_error": 200, "latency_ms": 200},
    ]

    report = build_report(evaluations)

    assert report["within_range_rate"] == 0.5
    assert report["mean_absolute_error"] == 105
    assert report["mean_latency_ms"] == 150


def test_build_report_abstention_correct_is_true_only_if_all_safety_cases_pass():
    evaluations = [
        {"type": "abstention", "passed": True, "latency_ms": 50},
        {"type": "partial_precedent", "passed": False, "latency_ms": 50},
    ]

    report = build_report(evaluations)

    assert report["abstention_correct"] is False


def test_build_report_handles_no_estimation_cases_without_crashing():
    evaluations = [{"type": "abstention", "passed": True, "latency_ms": 50}]

    report = build_report(evaluations)

    assert report["within_range_rate"] is None
    assert report["mean_absolute_error"] is None
    assert report["abstention_correct"] is True


# ---------------------------------------------------------------------------
# golden_set.json -- structural sanity (still no network/model call)
# ---------------------------------------------------------------------------


def test_golden_set_has_at_least_five_cases_including_one_abstention():
    cases = load_golden_set()

    assert len(cases) >= 5
    assert any(c["case_type"] == "abstention" for c in cases)


def test_golden_set_estimation_cases_have_a_range_containing_the_expected_value():
    cases = load_golden_set()

    for case in cases:
        if case["case_type"] != "estimation":
            continue
        low, high = case["acceptable_range"]
        assert low <= case["expected_hours"] <= high
