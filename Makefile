all:
	cat Makefile

test:
	pytest mutual_info

install:
	python setup.py install

develop:
	python setup.py develop

autoformat:
	find . -name '*.py' | xargs ./scripts/ap300.bash
