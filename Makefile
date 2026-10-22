PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: setup migrate load-data train test up down

setup:
	$(PIP) install -e ".[dev]"
	pre-commit install

migrate:
	alembic upgrade head

load-data: migrate
	$(PYTHON) -m recsys.data.loader --config configs/data.yaml

train:
	$(PYTHON) -m recsys.training --data-config configs/data.yaml --model-config configs/models.yaml --experiment-config configs/experiments.yaml

test:
	$(PYTHON) -m pytest

up:
	docker compose up --build -d

down:
	docker compose down
