FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY alembic.ini ./
COPY migrations/ ./migrations/

RUN python -m pip install --upgrade pip \
    && python -m pip install .

EXPOSE 8000 8501 5000

CMD ["uvicorn", "recsys.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
