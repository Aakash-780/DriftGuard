.PHONY: install train api dashboard mlflow test experiments experiments-quick report simulate reset
install:            ; pip install -r requirements.txt
train:              ; python -m driftguard.cli train
simulate:           ; python -m driftguard.cli simulate --scenario sudden
api:                ; uvicorn driftguard.app.main:app --port 8000
dashboard:          ; streamlit run driftguard/dashboard/app.py
mlflow:             ; mlflow ui --backend-store-uri sqlite:///$(CURDIR)/artifacts/mlflow/mlflow.db --port 5000
test:               ; python -m pytest
experiments:        ; python -m driftguard.experiments.run_all
experiments-quick:  ; python -m driftguard.experiments.run_all --quick --out artifacts/experiments_quick
report:             ; python -m driftguard.experiments.make_report
reset:              ; python -m driftguard.cli reset --yes
