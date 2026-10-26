from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pandas as pd
import pytest

from recsys.ui import app as ui


def _run(run_id: str, name: str, validation_ndcg: float) -> SimpleNamespace:
    return SimpleNamespace(
        info=SimpleNamespace(run_name=name, run_id=run_id),
        data=SimpleNamespace(
            metrics={
                "validation_ndcg@10": validation_ndcg,
                "test_ndcg@10": validation_ndcg - 0.02,
            }
        ),
    )


def test_model_comparison_frame_sorts_by_validation_ndcg() -> None:
    frame = ui.model_comparison_frame(
        [
            _run("first", "popularity", 0.2),
            _run("second", "hybrid", 0.4),
        ]
    )

    assert frame["model"].tolist() == ["hybrid", "popularity"]
    assert frame.iloc[0]["validation_ndcg@10"] == pytest.approx(0.4)
    assert frame.iloc[0]["test_ndcg@10"] == pytest.approx(0.38)


def test_fetch_api_json_decodes_object_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"experiment_name": "demo"}).encode()

    monkeypatch.setattr(ui, "urlopen", lambda request, timeout: FakeResponse())

    assert ui.fetch_api_json("/recommend/7?k=2") == {"experiment_name": "demo"}


def test_fetch_api_json_reports_upstream_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_http_error(request: object, timeout: int) -> None:
        raise HTTPError(
            "http://api/experiments/demo/results",
            503,
            "Unavailable",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(ui, "urlopen", raise_http_error)
    with pytest.raises(ui.UIServiceError, match="HTTP 503"):
        ui.fetch_api_json("/experiments/demo/results")

    monkeypatch.setattr(
        ui,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(URLError("offline")),
    )
    with pytest.raises(ui.UIServiceError, match="Could not reach"):
        ui.fetch_api_json("/recommend/1")

    monkeypatch.setattr(
        ui,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(TimeoutError()),
    )
    with pytest.raises(ui.UIServiceError, match="timed out"):
        ui.fetch_api_json("/recommend/1")


def test_load_model_comparison_queries_mlflow_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, tracking_uri: str) -> None:
            assert tracking_uri == ui.MLFLOW_TRACKING_URI

        def get_experiment_by_name(self, name: str) -> SimpleNamespace:
            assert name == "movielens-25m"
            return SimpleNamespace(experiment_id="experiment-id")

        def search_runs(
            self, experiment_ids: list[str], order_by: list[str], max_results: int
        ) -> list[SimpleNamespace]:
            assert experiment_ids == ["experiment-id"]
            assert order_by == ["start_time DESC"]
            assert max_results == 100
            return [_run("run-1", "hybrid", 0.5)]

    monkeypatch.setattr(ui, "MlflowClient", FakeClient)
    monkeypatch.setattr(ui.mlflow, "set_tracking_uri", lambda uri: None)

    frame = ui.load_model_comparison("movielens-25m")

    assert frame["model"].tolist() == ["hybrid"]
    assert frame.iloc[0]["validation_ndcg@10"] == pytest.approx(0.5)


def test_model_comparison_frame_has_columns_when_empty() -> None:
    frame = ui.model_comparison_frame([])

    assert isinstance(frame, pd.DataFrame)
    assert frame.empty
    assert list(frame.columns) == []
