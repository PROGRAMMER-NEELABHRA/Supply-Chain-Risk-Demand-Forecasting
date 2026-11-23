from types import SimpleNamespace

import run_pipeline


def test_pipeline_stops_and_reports_failed_stage(monkeypatch, capsys):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(run_pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_pipeline,
        "STAGES",
        [
            ("FIRST STAGE", ["first"]),
            ("SECOND STAGE", ["second"]),
        ],
    )

    result = run_pipeline.main()

    assert result == 2
    assert calls == [["first"]]
    assert "PIPELINE FAILED AT: FIRST STAGE" in capsys.readouterr().out
