import pandas as pd
import json
import os
import time
import random
from datetime import datetime
from jobspy import scrape_jobs

SEARCH_QUERIES = [
    {"site_name": ["linkedin", "indeed"], "search_term": "software engineer intern", "location": "India", "results_wanted": 20},
    {"site_name": ["linkedin", "indeed"], "search_term": "data analyst intern", "location": "India", "results_wanted": 20},
    {"site_name": ["linkedin", "indeed"], "search_term": "backend developer intern", "location": "India", "results_wanted": 20},
    {"site_name": ["linkedin", "indeed"], "search_term": "frontend developer intern", "location": "India", "results_wanted": 15},
    {"site_name": ["linkedin", "indeed"], "search_term": "machine learning intern", "location": "India", "results_wanted": 15},
]

KEEP_COLUMNS = [
    "id",
    "site",
    "job_url",
    "job_url_direct",
    "title",
    "company",
    "location",
    "date_posted",
    "job_type",
    "salary_source",
    "interval",
    "min_amount",
    "max_amount",
    "currency",
    "is_remote",
    "job_level",
    "job_function",
    "listing_type",
    "emails",
    "description",
    "company_industry",
    "company_url",
    "company_logo",
    "company_url_direct",
    "company_addresses",
    "company_num_employees",
    "company_revenue",
    "company_description",
    "skills",
    "experience_range",
    "company_rating",
    "company_reviews_count",
    "vacancy_count",
    "work_from_home_type",
]


def scrape_jobs_list() -> list[dict]:
    """Scrape jobs from multiple ATS platforms using jobspy."""
    all_frames = []

    for query in SEARCH_QUERIES:
        search_term = query["search_term"]
        print(f"Scraping: '{search_term}' ...")
        try:
            kwargs = {}
            if "linkedin" in query["site_name"]:
                kwargs["linkedin_fetch_description"] = True

            df = scrape_jobs(
                site_name=query["site_name"],
                search_term=search_term,
                location=query["location"],
                results_wanted=query["results_wanted"],
                **kwargs,
            )
            print(f"  Found {len(df)} results")
            all_frames.append(df)
        except Exception as e:
            print(f"  Failed to scrape '{search_term}': {e}")

        time.sleep(random.uniform(3, 7))

    if not all_frames:
        print("No results scraped from any query.")
        return []

    combined = pd.concat(all_frames, ignore_index=True)

    # Remove duplicates based on job_url
    if "job_url" in combined.columns:
        combined = combined.drop_duplicates(subset="job_url", keep="first")

    # Keep only desired columns, fill missing with empty string
    for col in KEEP_COLUMNS:
        if col not in combined.columns:
            combined[col] = ""
    combined = combined[KEEP_COLUMNS].fillna("")

    # Convert all values to strings for JSON serialization
    combined = combined.astype(str)

    jobs = combined.to_dict(orient="records")
    return jobs


def save_scraped_jobs(jobs: list[dict]) -> None:
    """Save scraped jobs to JSON and CSV with a timestamp."""
    os.makedirs("data", exist_ok=True)

    # Add scrape timestamp
    payload = {
        "scraped_at": datetime.now().isoformat(),
        "count": len(jobs),
        "jobs": jobs,
    }

    with open("data/scraped_jobs.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    pd.DataFrame(jobs).to_csv("data/scraped_jobs.csv", index=False)

    print(f"Saved {len(jobs)} jobs to data/scraped_jobs.json")


def load_scraped_jobs() -> list[dict]:
    """Load scraped jobs from disk, or scrape fresh if file doesn't exist."""
    json_path = "data/scraped_jobs.json"
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("jobs", [])

    # File doesn't exist — scrape, save, then return
    jobs = scrape_jobs_list()
    save_scraped_jobs(jobs)
    return jobs


if __name__ == "__main__":
    jobs = scrape_jobs_list()
    save_scraped_jobs(jobs)
    print(f"\nScraped {len(jobs)} jobs total")
    print("Sample job:")
    print(jobs[0] if jobs else "No jobs found")
