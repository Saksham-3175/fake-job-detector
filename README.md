# Fake Job Detector
Detects fake job listings using ML. Scrapes real jobs from ATS platforms,
classifies them as real or fake, and displays verified listings with apply links.

## Setup
pip install -r requirements.txt

## Run Order
1. python ml/preprocess.py
2. python ml/train.py
3. python scraper/job_scraper.py
4. streamlit run ui/app.py
