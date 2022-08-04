.DEFAULT_GOAL:= clean-install
.PHONY: clean install test clean-install

clean:
	rm -fr build/
	rm -fr dist/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '__pycache__' -exec rm -fr {} +

install:
	pip install .

test:
	python -m unittest discover tests

clean-install: install clean