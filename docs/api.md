# FastAPI serving

The API starts with `uvicorn recsys.api.app:app`. It uses `DATABASE_URL` for
SQLAlchemy and `MLFLOW_TRACKING_URI` to resolve model artifacts. Set
`RECSYS_CONTROL_MODEL_URI` and `RECSYS_TREATMENT_MODEL_URI` to MLflow model URIs
(for example, `runs:/<run-id>/model`) to serve the configured control and
treatment models. Optionally set `RECSYS_SIMILAR_MODEL_URI` to an item-kNN model
artifact for `/similar/{movie_id}`. Model URIs are passed into the API
container by Compose. If a recommendation or similarity model is not configured,
that route returns HTTP 503 rather than silently returning a fallback.

Run `alembic upgrade head` before serving so the `experiments` and `feedback`
tables exist. The API container's Compose startup command runs this migration
before Uvicorn. The routes do not fetch or load the MovieLens source CSVs.

Endpoints:

- `GET /recommend/{user_id}?k=10` assigns a stable configured experiment group
  and returns ranked recommendations with available movie metadata.
- `GET /similar/{movie_id}?k=10` returns nearest neighbors from the configured
  item-kNN artifact.
- `GET /movies/{movie_id}` returns catalog metadata.
- `POST /feedback` persists a rating or click event.
- `GET /experiments/{name}/results` returns persisted offline-simulation
  results. These are not live A/B measurements.

Swagger UI is available at `/docs`.

## Streamlit

The UI starts with `streamlit run src/recsys/ui/app.py`. Configure
`API_BASE_URL` to point to the FastAPI service and `MLFLOW_TRACKING_URI` to the
same MLflow tracking server/store used for training. The app provides
recommendation lookup by MovieLens user ID, model-metric comparison from the
`movielens-25m` MLflow experiment, and stored offline A/B simulation results.
The model-comparison experiment can be changed with `MLFLOW_EXPERIMENT_NAME`;
the A/B experiment name can be changed with `RECSYS_EXPERIMENT_NAME`.
