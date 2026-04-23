.PHONY: install test clean-data

install:
	pip install -e ".[metrics,research,dev]"

test:
	pytest helium/ -v

# Wipe all job data (destructive!)
clean-data:
	rm -rf ~/.helium/jobs/*
	@echo "Job data cleared."
