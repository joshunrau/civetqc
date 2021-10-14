.DEFAULT_GOAL:= clean-install
.PHONY: clean install clean-install
.ONESHELL:

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

install:
	@type virtualenv >/dev/null 2>&1 || pip install virtualenv
	virtualenv --no-download venv
	(source venv/bin/activate)
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	pip install .

clean-install: clean install