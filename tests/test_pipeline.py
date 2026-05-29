"""End-to-end smoke test for the top-level training pipeline (``run_pipeline.py``).

The pipeline is the single command a reviewer runs to reproduce every reported
number, so this test verifies that one fast, reduced run actually produces the
three deliverables the project promises:

1. a trained model artifact (joblib),
2. the generated dataset CSV, and
3. ``reports/evaluation_report.md`` populated with REAL computed numbers
   (a model-vs-baseline comparison table and the failure-mode tables), plus
   the diagnostic figures it links to.

WHY this is asserted the way it is (intent, not just behaviour):

* A pipeline that "runs" but writes no model or an empty report is a silent
  failure -- the whole point is reproducible artifacts, so we assert the files
  exist AND are non-trivial.
* We assert the report contains the literal headline numbers returned by
  ``run()`` (e.g. the held-out MAE), so a report that hard-codes or drops the
  real metrics would fail. The report must reflect *this run*.
* We assert the comparison table and failure-mode section headers are present,
  because the case study explicitly requires a baseline comparison and a
  "where it breaks" analysis; their absence is a regression even if numbers
  look fine.

Speed/isolation: every canonical path and the sample count are monkeypatched
onto the ``run_pipeline`` module so the test trains on a few hundred rows and
writes only into ``tmp_path`` -- it never touches the repository's real
``data/``, ``models/`` or ``reports/`` artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import run_pipeline


@pytest.fixture
def fast_pipeline_paths(tmp_path, monkeypatch):
    """Redirect every pipeline output into tmp_path and shrink the dataset.

    ``run_pipeline`` reads its configuration from module-level constants that
    were bound at import time, so we patch the module attributes directly.
    Returns the tmp paths so the test can assert against them.
    """
    dataset_path = tmp_path / "dataset.csv"
    model_path = tmp_path / "energy_model.joblib"
    reports_dir = tmp_path / "reports"
    figures_dir = reports_dir / "figures"
    report_path = reports_dir / "evaluation_report.md"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Small but real: enough rows to fit HGB + linear and slice failure modes.
    monkeypatch.setattr(run_pipeline, "N_SAMPLES", 500, raising=True)
    monkeypatch.setattr(run_pipeline, "SEED", 42, raising=True)
    monkeypatch.setattr(run_pipeline, "TEST_SIZE", 0.25, raising=True)

    monkeypatch.setattr(run_pipeline, "DEFAULT_DATASET_PATH", dataset_path, raising=True)
    monkeypatch.setattr(run_pipeline, "DEFAULT_MODEL_PATH", model_path, raising=True)
    monkeypatch.setattr(run_pipeline, "REPORTS_DIR", reports_dir, raising=True)
    monkeypatch.setattr(run_pipeline, "REPORT_PATH", report_path, raising=True)

    # make_plots defaults to REPORTS_DIR/"figures"; force it onto our tmp dir so
    # figures land beside the report and the report's relative links resolve.
    from nexdash import evaluate as evaluate_module

    _orig_make_plots = evaluate_module.make_plots

    def _make_plots_tmp(model, df_test, out_dir=figures_dir):
        return _orig_make_plots(model, df_test, out_dir=out_dir)

    monkeypatch.setattr(run_pipeline.evaluate, "make_plots", _make_plots_tmp, raising=True)

    return {
        "dataset": dataset_path,
        "model": model_path,
        "reports_dir": reports_dir,
        "figures_dir": figures_dir,
        "report": report_path,
    }


def test_pipeline_produces_artifacts(fast_pipeline_paths):
    """A reduced end-to-end run writes the dataset, model and report artifacts."""
    result = run_pipeline.run()

    dataset_path = fast_pipeline_paths["dataset"]
    model_path = fast_pipeline_paths["model"]
    report_path = fast_pipeline_paths["report"]

    # 1. Dataset CSV exists and is non-empty.
    assert dataset_path.exists(), "pipeline did not write the dataset CSV"
    assert dataset_path.stat().st_size > 0

    # 2. Model artifact exists and is non-trivial (a real joblib payload).
    assert model_path.exists(), "pipeline did not save the model artifact"
    assert model_path.stat().st_size > 1024

    # 3. Evaluation report exists and is substantial.
    assert report_path.exists(), "pipeline did not write evaluation_report.md"
    report_text = report_path.read_text(encoding="utf-8")
    assert len(report_text) > 500

    # The returned object exposes the key results for programmatic callers.
    assert result["report_path"] == report_path
    assert "metrics" in result and "mae_kwh" in result["metrics"]
    assert result["model"] is not None


def test_report_contains_real_numbers_and_required_sections(fast_pipeline_paths):
    """The report must embed THIS run's real metrics, comparison & failure tables."""
    result = run_pipeline.run()
    report_text = fast_pipeline_paths["report"].read_text(encoding="utf-8")

    # Required section structure (baseline comparison + failure-mode analysis).
    assert "Model vs. linear baseline" in report_text
    assert "HistGradientBoosting" in report_text
    assert "LinearRegression" in report_text
    assert "Failure-mode analysis" in report_text
    assert "By temperature" in report_text
    assert "By gradient" in report_text
    assert "By payload" in report_text

    # The headline MAE returned by run() must appear (formatted to 3 dp) in the
    # report -- proving the report reflects the numbers computed this run, not a
    # stale/hard-coded value.
    mae = float(result["metrics"]["mae_kwh"])
    assert f"{mae:.3f}" in report_text

    # The report must contain a markdown metric table (the comparison table).
    assert "| Metric |" in report_text
    assert "MAE (kWh)" in report_text


def test_pipeline_generates_diagnostic_figures(fast_pipeline_paths):
    """The pipeline must render diagnostic figures and link them from the report."""
    result = run_pipeline.run()

    figures = result["figures"]
    assert isinstance(figures, list) and len(figures) >= 1
    for fig in figures:
        fig_path = Path(fig)
        assert fig_path.exists(), f"figure not written: {fig_path}"
        assert fig_path.stat().st_size > 0
        # Figures must live under the tmp figures dir, proving isolation worked.
        assert fig_path.parent == fast_pipeline_paths["figures_dir"]
