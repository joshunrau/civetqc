.DEFAULT_GOAL:= clean-install
.PHONY: clean check_env install test clean-install

clean:
	rm -fr build/
	rm -fr dist/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '__pycache__' -exec rm -fr {} +

check_env:
	@if [ $${CONDA_PREFIX:(-7):7} != civetqc ]; then echo ERROR: civetqc environment not set && exit 1; fi

install: check_env
	pip install .

install-dev: check_env
	pip install --editable .

test:
	python -m unittest discover tests

clean-install: install clean