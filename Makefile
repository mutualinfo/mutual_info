all:
	cat Makefile

test:
	pytest mutual_info

install:
	python setup.py install

develop:
	python setup.py develop

autoformat:
	./scripts/ap300.bash $$(find . -name '*.py')
