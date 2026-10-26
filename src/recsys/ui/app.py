"""Streamlit interface for recommendations, MLflow comparisons, and offline tests."""

from __future__ import annotations

import json
import os
from json import JSONDecodeError
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
DEFAULT_EXPERIMENT_NAME = os.getenv(
    "RECSYS_EXPERIMENT_NAME", "recommendation-ranking-v1"
)
DEFAULT_MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "movielens-25m")


class UIServiceError(RuntimeError):
    """An upstream API or MLflow service returned an unusable response."""


def fetch_api_json(path: str) -> dict[str, Any]:
    """Fetch one JSON response from the configured recommendation API."""
    request = Request(f"{API_BASE_URL}/{path.lstrip('/')}", method="GET")
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        raise UIServiceError(
            f"Recommendation API returned HTTP {error.code}: {response_body}"
        ) from error
    except URLError as error:
        raise UIServiceError(
            f"Could not reach the recommendation API: {error}"
        ) from error
    except TimeoutError as error:
        raise UIServiceError("Recommendation API request timed out") from error
    except (JSONDecodeError, UnicodeDecodeError) as error:
        raise UIServiceError("Recommendation API returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise UIServiceError("Recommendation API response must be a JSON object")
    return payload


def model_comparison_frame(runs: list[Run], metric_prefix: str = "") -> pd.DataFrame:
    """Build a sorted, display-ready dataframe from MLflow runs."""
    records: list[dict[str, Any]] = []
    for run in runs:
        metrics = {
            name.replace("_at_", "@"): value
            for name, value in run.data.metrics.items()
            if name.startswith(metric_prefix)
        }
        records.append(
            {
                "model": run.info.run_name or run.info.run_id,
                "run_id": run.info.run_id,
                **metrics,
            }
        )
    frame = pd.DataFrame(records)
    validation_ndcg_columns = sorted(
        column for column in frame.columns if column.startswith("validation_ndcg@")
    )
    if validation_ndcg_columns:
        frame = frame.sort_values(
            validation_ndcg_columns[0], ascending=False, na_position="last"
        )
    return frame.reset_index(drop=True)


def load_model_comparison(experiment_name: str) -> pd.DataFrame:
    """Read the experiment's latest model runs and logged evaluation metrics."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame(columns=["model", "run_id"])
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=100,
    )
    return model_comparison_frame(runs)


def _render_recommendations() -> None:
    st.header("Movie recommendations")
    st.write("Choose a MovieLens user ID to request ranked recommendations.")
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    count = st.slider("Number of recommendations", min_value=1, max_value=50, value=10)
    if st.button("Get recommendations", type="primary"):
        try:
            result = fetch_api_json(f"/recommend/{int(user_id)}?k={count}")
        except UIServiceError as error:
            st.error(str(error))
            return
        st.caption(
            f"Experiment: {result['experiment_name']} · Group: {result['group']}"
        )
        recommendations = result.get("recommendations", [])
        if not recommendations:
            st.info("No unseen recommendations are available for this user.")
            return
        for recommendation in recommendations:
            title = recommendation.get("title") or f"Movie {recommendation['movie_id']}"
            with st.container(border=True):
                st.subheader(title)
                st.caption(
                    f"Movie ID: {recommendation['movie_id']} · "
                    f"Genres: {recommendation.get('genres') or 'Not available'}"
                )
                st.write(recommendation["explanation"])


def _render_model_comparison() -> None:
    st.header("Model comparison")
    experiment_name = st.text_input(
        "MLflow experiment", value=DEFAULT_MLFLOW_EXPERIMENT
    ).strip()
    if st.button("Load model metrics", type="primary"):
        if not experiment_name:
            st.error("Enter an MLflow experiment name.")
            return
        try:
            comparison = load_model_comparison(experiment_name)
        except mlflow.exceptions.MlflowException as error:
            st.error(f"Could not read MLflow runs: {error}")
            return
        if comparison.empty or len(comparison.columns) <= 2:
            st.info("No logged model metrics were found for this experiment.")
            return
        st.dataframe(comparison.drop(columns=["run_id"]), use_container_width=True)
        metric_columns = [
            column
            for column in comparison.columns
            if column.startswith(("validation_", "test_"))
        ]
        metric = st.selectbox("Metric chart", options=metric_columns)
        chart_data = comparison[["model", metric]].dropna()
        if not chart_data.empty:
            st.plotly_chart(
                px.bar(
                    chart_data,
                    x="model",
                    y=metric,
                    title=f"{metric} by model",
                    labels={"model": "Model", metric: metric},
                ),
                use_container_width=True,
            )


def _render_ab_results() -> None:
    st.header("Offline A/B simulation results")
    st.warning(
        "These results are offline simulations. Historical ratings are click proxies; "
        "they are not live-user A/B measurements."
    )
    experiment_name = st.text_input(
        "Experiment name", value=DEFAULT_EXPERIMENT_NAME
    ).strip()
    if st.button("Load experiment results", type="primary"):
        if not experiment_name:
            st.error("Enter an experiment name.")
            return
        try:
            payload = fetch_api_json(
                f"/experiments/{quote(experiment_name, safe='')}/results"
            )
        except UIServiceError as error:
            st.error(str(error))
            return
        results = payload.get("results", [])
        if not results:
            st.info("No saved results exist for this experiment.")
            return
        frame = pd.DataFrame(results)
        display_columns = [
            "created_at",
            "control_name",
            "treatment_name",
            "control_users",
            "treatment_users",
            "control_ctr",
            "treatment_ctr",
            "effect",
            "p_value",
            "confidence_interval_low",
            "confidence_interval_high",
            "sample_size_per_group",
            "statistical_test",
            "cuped_applied",
            "is_simulated",
        ]
        st.dataframe(frame[display_columns], use_container_width=True)
        latest = results[0]
        left, right = st.columns(2)
        left.metric("Control CTR (proxy)", f"{latest['control_ctr']:.3%}")
        right.metric("Treatment CTR (proxy)", f"{latest['treatment_ctr']:.3%}")
        chart_frame = pd.DataFrame(
            [
                {
                    "group": latest["control_name"],
                    "proxy CTR": latest["control_ctr"],
                },
                {
                    "group": latest["treatment_name"],
                    "proxy CTR": latest["treatment_ctr"],
                },
            ]
        )
        st.plotly_chart(
            px.bar(chart_frame, x="group", y="proxy CTR", title="Proxy CTR by group"),
            use_container_width=True,
        )
        st.caption(
            f"p-value: {latest['p_value']:.4g} · "
            f"95% bootstrap CI for treatment-control effect: "
            f"[{latest['confidence_interval_low']:.4g}, "
            f"{latest['confidence_interval_high']:.4g}]"
        )


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(page_title="MovieLens Recommender", layout="wide")
    st.title("MovieLens Recommender")
    page = st.sidebar.radio(
        "Page",
        (
            "Recommendations",
            "Model comparison",
            "A/B results",
        ),
    )
    if page == "Recommendations":
        _render_recommendations()
    elif page == "Model comparison":
        _render_model_comparison()
    else:
        _render_ab_results()


if __name__ == "__main__":
    main()
