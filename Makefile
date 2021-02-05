all:
	cat Makefile

test:
	pytest mutual_info

install:
	python setup.py install

develop:
	python setup.py develop
