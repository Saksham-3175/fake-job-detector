# NOTE: models/fake_job_model.joblib must exist before building this image.
# Run `make train` (or `python ml/train.py`) first to generate the model file.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
