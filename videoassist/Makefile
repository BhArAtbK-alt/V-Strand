.PHONY: install data eval demo test api
install: ; pip install -r requirements.txt && pip install -e .
data:    ; python scripts/make_dataset.py 60
eval:    ; python scripts/run_eval.py
demo:    ; python scripts/demo.py
test:    ; pytest -q
api:     ; uvicorn videoassist.api:app --reload
