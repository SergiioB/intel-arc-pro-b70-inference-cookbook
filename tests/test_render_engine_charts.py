"""Tests for scripts/render-engine-charts.py — renders committed SVGs from
data/chart-inputs.v1.json; --check verifies they match."""
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render-engine-charts.py"
INPUT = ROOT / "data" / "chart-inputs.v1.json"

spec = importlib.util.spec_from_file_location("render_engine_charts", SCRIPT)
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)


def run(args):
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + args, capture_output=True, text=True)


def base_chart():
    return {
        "id": "t",
        "output": "docs/assets/t.svg",
        "engine": "vLLM XPU",
        "engine_color": "vllm",
        "eyebrow": "e",
        "title": "t",
        "subtitle": "s",
        "y_label": "y",
        "x_label": "x",
        "x_axis": {"min": 512, "max": 8192},
        "x_ticks": [{"value": 512, "label": "512"}, {"value": 8192, "label": "8K"}],
        "series": [{"label": "s1", "color": "vllm",
                    "segments": [{"style": "solid", "protocol": "p",
                                  "points": {"512": 20.0, "8192": 18.0}}]}],
    }


def test_load_inputs_reads_committed_file():
    charts = rc.load_inputs(INPUT)
    assert len(charts) >= 5


def test_load_inputs_rejects_missing_charts(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "1.0"}))
    with pytest.raises(SystemExit):
        rc.load_inputs(bad)


def test_validate_accepts_wellformed():
    rc.validate_chart(base_chart(), 0)


@pytest.mark.parametrize("field", ["id", "output", "engine", "engine_color",
                                   "eyebrow", "title", "subtitle", "y_label", "x_label"])
def test_validate_requires_fields(field):
    c = base_chart()
    del c[field]
    with pytest.raises(SystemExit):
        rc.validate_chart(c, 0)


def test_validate_rejects_bad_axis():
    c = base_chart()
    c["x_axis"] = {"min": 8192, "max": 512}
    with pytest.raises(SystemExit):
        rc.validate_chart(c, 0)


def test_validate_rejects_unknown_engine_color():
    c = base_chart()
    c["engine_color"] = "not-an-engine"
    with pytest.raises(SystemExit):
        rc.validate_chart(c, 0)


def test_validate_rejects_series_color_mismatch():
    c = base_chart()
    c["series"][0]["color"] = "ovms"
    with pytest.raises(SystemExit):
        rc.validate_chart(c, 0)


def test_validate_rejects_bad_segment_style():
    c = base_chart()
    c["series"][0]["segments"][0]["style"] = "wavy"
    with pytest.raises(SystemExit):
        rc.validate_chart(c, 0)


def test_validate_rejects_point_outside_axis():
    c = base_chart()
    c["series"][0]["segments"][0]["points"]["99999999"] = 12.3
    with pytest.raises(SystemExit):
        rc.validate_chart(c, 0)


def test_render_chart_emits_valid_svg():
    c = base_chart()
    c["x_ticks"] = [{"value": 512, "label": "512"}, {"value": 2048, "label": "2K"},
                    {"value": 8192, "label": "8K"}]
    svg = rc.render_chart(c)
    assert svg.startswith("<svg") or svg.lstrip().startswith("<svg")
    assert "PROVISIONAL" in svg or "t" in svg
    assert "20.0" in svg


def test_sparse_label_set_endpoints():
    seg = {"points": {"512": 1.0, "1024": 2.0, "4096": 3.0, "8192": 4.0, "16384": 5.0}}
    pts = rc.sorted_points(seg)
    ends = rc.sparse_label_set(pts, 512, 16384, "ends")
    assert ends == {512, 16384}


def test_sorted_points_orders_keys():
    seg = {"points": {"8192": 1.0, "512": 2.0, "4096": 3.0}}
    assert rc.sorted_points(seg) == [(512, 2.0), (4096, 3.0), (8192, 1.0)]


def test_check_mode_verifies_committed_svgs():
    completed = run(["--check"])
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Charts valid and current" in completed.stdout


def test_render_writes_svgs_to_staging(tmp_path):
    staging = ROOT / ".ci-chart-staging"
    try:
        data = json.loads(INPUT.read_text())
        for chart in data["charts"]:
            chart["output"] = str(staging / pathlib.Path(chart["output"]).name)
        staging_input = tmp_path / "inputs.json"
        staging_input.write_text(json.dumps(data))
        completed = run(["--input", str(staging_input)])
        assert completed.returncode == 0, completed.stderr
        rendered = list(staging.glob("*.svg"))
        assert len(rendered) == len(data["charts"])
        for svg in rendered:
            assert "PROVISIONAL" in svg.read_text()
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def test_check_fails_on_tampered_svg(tmp_path):
    staging = ROOT / ".ci-chart-staging"
    try:
        data = json.loads(INPUT.read_text())
        first = data["charts"][0]
        data["charts"] = [first]
        out = staging / pathlib.Path(first["output"]).name
        first["output"] = str(out)
        staging_input = tmp_path / "inputs.json"
        staging_input.write_text(json.dumps(data))
        assert run(["--input", str(staging_input)]).returncode == 0
        out.write_text(out.read_text() + "<!-- tampered -->")
        completed = run(["--input", str(staging_input), "--check"])
        assert completed.returncode != 0
    finally:
        shutil.rmtree(staging, ignore_errors=True)
