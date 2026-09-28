"""Harness aggregation: ASR, benign success, latency robustness, and configurations."""

from __future__ import annotations

from src.eval.harness import (
    Configuration, HarnessResult, RunRecord, default_benign_tasks, standard_configurations,
)
from src.firewall.pipeline import LayerSwitches


def _result() -> HarnessResult:
    r = HarnessResult()
    # 4 attacks per config, 2 categories; no_firewall: 3/4 succeed, guarded: 0/4.
    for config, successes in (("no_firewall", [1, 1, 1, 0]), ("all_layers", [0, 0, 0, 0])):
        for i, success in enumerate(successes):
            r.records.append(RunRecord(config, "attack", f"a{i}", "indirect" if i < 2 else "exfil_tool",
                                       seed=i, success=success, wall_time_s=1.0))
        for i in range(2):
            r.records.append(RunRecord(config, "benign", f"b{i}", "benign", seed=i,
                                       success=1, wall_time_s=1.0))
    return r


def test_asr_overall_and_by_category():
    r = _result()
    assert r.asr("no_firewall").rate == 0.75
    assert r.asr("all_layers").rate == 0.0
    assert r.asr("no_firewall", "indirect").rate == 1.0
    assert r.asr("no_firewall", "exfil_tool").rate == 0.5


def test_asr_has_confidence_interval():
    p = _result().asr("no_firewall")
    assert p.low < p.rate < p.high and p.total == 4


def test_benign_success_is_separate_from_asr():
    r = _result()
    assert r.benign_success("all_layers").rate == 1.0
    assert r.benign_success("all_layers").total == 2  # benign rows only


def test_median_latency_ignores_a_single_stalled_run():
    """One stalled model call must not be reported as firewall overhead."""
    r = _result()
    r.records.append(RunRecord("no_firewall", "attack", "stalled", "indirect", seed=9,
                               success=0, wall_time_s=4000.0))
    assert r.median_latency("no_firewall") == 1.0
    assert r.mean_latency("no_firewall") > 500  # the mean is destroyed by the outlier


def test_configs_preserve_insertion_order():
    assert _result().configs() == ["no_firewall", "all_layers"]


def test_standard_configurations_cover_baseline_each_layer_and_full():
    names = [c.name for c in standard_configurations()]
    assert names[0] == "no_firewall"
    assert all(f"only_{layer}" in names for layer in LayerSwitches.NAMES)
    assert "all_layers" in names and "all_layers_block" in names


def test_baseline_configuration_is_undefended():
    baseline = standard_configurations()[0]
    assert not baseline.defended and baseline.switches is None


def test_benign_tasks_have_checkable_goals():
    tasks = default_benign_tasks()
    assert len(tasks) >= 4
    assert all(callable(t.check) and t.task for t in tasks)


def test_configuration_defended_flag():
    assert Configuration("x", LayerSwitches()).defended
    assert not Configuration("y", None).defended
