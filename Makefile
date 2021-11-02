.DEFAULT_GOAL:= clean-install
.PHONY: clean requirements install test clean-install

clean:
	rm -fr build/
	rm -fr dist/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '__pycache__' -exec rm -fr {} +

requirements:
	pipreqs . --force

install:
	@if [ $${CONDA_PREFIX:(-7):7} != civetqc ]; then echo ERROR: must install in civetqc environment && exit 1; fi
	pip install -r requirements.txt
	pip install .

st:
	@if [ $${CONDA_PREFIX:(-7):7} != civetqc ]; then echo ERROR: must install in civetqc environment && exit 1; fi
	./bin/shell_test

test:
	python -m unittest discover .

clean-install: clean install
