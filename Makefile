PY_PROJECT_DIR := node2vec

.PHONY: install
install:  ## Install development dependencies.
	pip install -r requirements.txt

.PHONY: tests
tests:  ## Run the tests.
	pytest tests/ -v

.PHONY: format
format:  ## Format the code.
	isort -rc ${PY_PROJECT_DIR}
	isort -rc tests/
	black ${PY_PROJECT_DIR}

.PHONY: checks
checks:  ## Run static analyses of the code.
	mypy ${PY_PROJECT_DIR}
	flake8 ${PY_PROJECT_DIR}

.PHONY: wheel
wheel:	## Create a wheel package.
	python setup.py bdist_wheel

.PHONY: clean
clean:	## Remove artefacts.
	rm -rf build/ tmp/ *.egg-info/ dist/

.PHONY: help
help:	## Display this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
