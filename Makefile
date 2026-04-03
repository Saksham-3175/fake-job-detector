.PHONY: preprocess train promote serve ui docker test lint clean all

preprocess:
	python ml/preprocess.py

train:
	python ml/train.py

promote:
	python ml/train.py --promote

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run ui/app.py --server.port 8501

docker:
	docker-compose up --build

test:
	pytest tests/ -v

lint:
	ruff check .

clean:
	rm -f data/train.csv data/test.csv data/scraped_jobs.json data/scraped_jobs.csv \
		models/fake_job_model.joblib models/confusion_matrix.png
	rm -rf mlruns/ __pycache__/

all: preprocess train
