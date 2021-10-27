.DEFAULT_GOAL:= clean-install
.PHONY: help clean requirements install clean-install

help:
	@echo "clean - remove all build, test, coverage and Python artifacts"
	@echo "install - create new virtual environment with package and dependencies installed"
	@echo "clean-install - run both clean and install (default)"

clean:
	rm -fr build/
	rm -fr dist/
	rm -fr venv/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '__pycache__' -exec rm -fr {} +

requirements:
	pipreqs . --force

test:
	python -m unittest discover .

shell_test:
	./bin/shell_test

install:
	pip install -r requirements.txt
	pip install .

clean-install: clean install