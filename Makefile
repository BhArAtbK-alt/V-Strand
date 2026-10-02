.PHONY: install demo train test
install:
	pip install -r requirements.txt && pip install -e .
demo:
	python scripts/run_demo.py
train:
	python -m vstrand.training.train_kg && python -m vstrand.training.train_approxbit
test:
	pytest -q
