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

clean:
	rm -rf dist build

pypi_test: clean
	python -m build
	# requires an api key
	python3 -m twine upload --repository testpypi dist/*

pypi: clean
	python -m build
	python3 -m twine upload dist/*

