import sys
import os

# Add project root to path so ml/ and scraper/ are importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml.predict import predict_listing
from scraper.job_scraper import load_scraped_jobs, scrape_jobs_list, save_scraped_jobs

app = FastAPI(title="Fake Job Detector API")

# Enable CORS (allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    title: str
    description: str
    company_profile: str = ""
    requirements: str = ""


def _enrich_job(job: dict) -> dict:
    """Run prediction on a single scraped job and return enriched dict."""
    result = predict_listing(
        title=job.get("title", ""),
        description=job.get("description", ""),
        company_profile=job.get("company", ""),
    )
    return {
        "title": job.get("title", ""),
        "company": job.get("company", ""),
        "location": job.get("location", ""),
        "job_type": job.get("job_type", ""),
        "apply_link": job.get("job_url", ""),
        "date_posted": job.get("date_posted", ""),
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "trust_level": result["trust_level"],
        "is_fake": result["is_fake"],
    }


@app.get("/")
def root():
    return {"status": "ok", "message": "Fake Job Detector API"}


@app.get("/jobs")
def get_jobs():
    scraped = load_scraped_jobs()
    enriched = [_enrich_job(job) for job in scraped]

    real_jobs = [j for j in enriched if not j["is_fake"]]
    fake_jobs = [j for j in enriched if j["is_fake"]]

    return {
        "total_scraped": len(enriched),
        "real_jobs_count": len(real_jobs),
        "fake_jobs_count": len(fake_jobs),
        "real_jobs": real_jobs,
        "fake_jobs": fake_jobs,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    result = predict_listing(
        title=req.title,
        description=req.description,
        company_profile=req.company_profile,
        requirements=req.requirements,
    )
    return result


@app.get("/jobs/refresh")
def refresh_jobs():
    jobs = scrape_jobs_list()
    save_scraped_jobs(jobs)
    return {"message": "Jobs refreshed", "count": len(jobs)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
