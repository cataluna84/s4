.PHONY: autoformat lint dev lib torchtext

.ONESHELL:

autoformat:
	black src/ checkpoints/ models/
	isort --atomic src/ checkpoints/ models/
	docformatter --in-place --recursive src checkpoints models

lint:
	isort -c src/ checkpoints/ models/
	black src/ checkpoints/ models/ --check
	flake8 src/ checkpoints/ models/

dev:
	pip install -r requirements-dev.txt

lib:
	pip install -r requirements.txt

torchtext: lib
	git clone https://github.com/pytorch/text build/torchtext
	cd build/torchtext
	git submodule update --init --recursive
	python setup.py clean install
