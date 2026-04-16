.PHONY: up down build logs shell test lint clean

up:
	docker-compose up --build

down:
	docker-compose down

build:
	docker-compose build

logs:
	docker-compose logs -f api

shell:
	docker-compose exec api bash

# Run tests inside the container
test:
	docker-compose exec api pytest tests/ -v

# Run tests locally (requires venv active)
test-local:
	cd backend && pytest tests/ -v

lint:
	cd backend && python -m py_compile $$(find app -name "*.py")

# Wipe all job data (destructive!)
clean-data:
	rm -rf data/jobs/*
	@echo "Job data cleared."
