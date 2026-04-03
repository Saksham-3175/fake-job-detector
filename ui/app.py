import os
import streamlit as st
import requests

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="Pipelined — Fake Job Detector",
    layout="wide",
    page_icon="🔍",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🔍 Pipelined")
    st.caption("AI-powered fake job detection")
    st.divider()

    st.subheader("Test a Listing")
    test_title = st.text_input("Job Title")
    test_description = st.text_area("Job Description", height=150)
    test_company = st.text_area("Company Info (optional)", height=100)

    if st.button("Analyze This Listing", use_container_width=True):
        if not test_title or not test_description:
            st.warning("Please enter at least a title and description.")
        else:
            try:
                resp = requests.post(
                    f"{API_BASE}/predict",
                    json={
                        "title": test_title,
                        "description": test_description,
                        "company_profile": test_company,
                        "requirements": "",
                    },
                    timeout=15,
                )
                result = resp.json()
                if result["verdict"] == "REAL":
                    st.success(
                        f"✅ **REAL** — Confidence: {result['confidence']:.0%}"
                    )
                else:
                    st.error(
                        f"⚠️ **FAKE** — Confidence: {result['confidence']:.0%}"
                    )
            except requests.exceptions.ConnectionError:
                st.error("Backend not running. Start it with:\n`uvicorn api.main:app --reload`")

# ── Refresh button (top-right) ───────────────────────────────────────────────

top_left, top_right = st.columns([5, 1])
with top_left:
    st.title("Pipelined — Fake Job Detector")
with top_right:
    st.markdown("")  # spacer
    if st.button("🔄 Refresh Jobs"):
        try:
            with st.spinner("Scraping latest jobs..."):
                resp = requests.get(f"{API_BASE}/jobs/refresh", timeout=120)
                data = resp.json()
            st.success(f"Jobs refreshed! {data['count']} listings scraped.")
        except requests.exceptions.ConnectionError:
            st.warning("Backend not running. Start it with: `uvicorn api.main:app --reload`")

# ── Fetch jobs from API ──────────────────────────────────────────────────────

try:
    jobs_resp = requests.get(f"{API_BASE}/jobs", timeout=15)
    jobs_data = jobs_resp.json()
except requests.exceptions.ConnectionError:
    st.warning("⚠️ Backend not running. Start it with: `uvicorn api.main:app --reload`")
    st.stop()
except Exception as e:
    st.error(f"Error fetching jobs: {e}")
    st.stop()

total = jobs_data.get("total_scraped", 0)
real_count = jobs_data.get("real_jobs_count", 0)
fake_count = jobs_data.get("fake_jobs_count", 0)
real_jobs = jobs_data.get("real_jobs", [])
fake_jobs = jobs_data.get("fake_jobs", [])

# ── Metric cards ─────────────────────────────────────────────────────────────

m1, m2, m3 = st.columns(3)
m1.metric("Total Jobs Scraped", total)
m2.metric("Verified Real", real_count)
m3.metric("Flagged Fake", fake_count)

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_real, tab_fake, tab_model = st.tabs(
    ["✅ Verified Jobs", "⚠️ Flagged Listings", "📊 Model Info"]
)

# ── Helper: render a job card ────────────────────────────────────────────────

def _trust_color(level: str) -> str:
    return {"High": "green", "Medium": "orange", "Low": "red"}.get(level, "gray")


def render_job_card(job: dict, *, flagged: bool = False):
    with st.container(border=True):
        # Row 1 — Title + trust badge
        trust = job.get("trust_level", "Low")
        color = _trust_color(trust)
        confidence_pct = f"{job.get('confidence', 0):.0%}"

        if flagged:
            badge = f":{color}[⚠️ {trust} confidence — {confidence_pct} fake probability]"
        else:
            badge = f":{color}[🛡️ {trust} confidence — {confidence_pct}]"

        st.markdown(f"### {job.get('title', 'Untitled')}  {badge}")

        # Row 2 — Company, location, type
        cols = st.columns(3)
        cols[0].markdown(f"🏢 **{job.get('company', 'N/A')}**")
        cols[1].markdown(f"📍 {job.get('location', 'N/A')}")
        cols[2].markdown(f"💼 {job.get('job_type', 'N/A')}")

        # Row 3 — Description snippet
        desc = job.get("description", "")
        if len(desc) > 200:
            desc = desc[:200] + "..."
        if desc:
            st.markdown(f"_{desc}_")

        # Row 4 — Date + Apply
        c_left, c_right = st.columns([4, 1])
        posted = job.get("date_posted", "")
        if posted:
            c_left.caption(f"📅 Posted: {posted}")
        apply_link = job.get("apply_link", "")
        if apply_link:
            c_right.link_button("Apply →", apply_link, use_container_width=True)


# ── Tab 1: Verified Jobs ────────────────────────────────────────────────────

with tab_real:
    st.header("Real Job Listings")
    st.caption(
        "These listings passed our ML verification. "
        "Click Apply to visit the original posting."
    )
    if real_jobs:
        for job in real_jobs:
            render_job_card(job)
    else:
        st.info("No verified jobs yet. Click **Refresh Jobs** to scrape listings.")

# ── Tab 2: Flagged Listings ─────────────────────────────────────────────────

with tab_fake:
    st.header("Flagged as Potentially Fake")
    st.caption("Our model flagged these listings. Proceed with caution.")
    if fake_jobs:
        for job in fake_jobs:
            render_job_card(job, flagged=True)
    else:
        st.info("No flagged listings found.")

# ── Tab 3: Model Info ───────────────────────────────────────────────────────

with tab_model:
    st.header("How The Model Works")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("📁 Dataset")
        st.markdown(
            "**EMSCAD** — 17,880 job listings\n\n"
            "- 17,014 real\n"
            "- 866 fake"
        )
    with c2:
        st.subheader("🤖 Model")
        st.markdown(
            "**TF-IDF Vectorizer**\n\n"
            "↓\n\n"
            "**Logistic Regression**\n\n"
            "(balanced class weights)"
        )
    with c3:
        st.subheader("🎯 Key Metric")
        st.markdown(
            "**Recall on fake listings**\n\n"
            "Measures how many actual fakes the model catches."
        )

    st.divider()

    # Confusion matrix image
    cm_path = "models/confusion_matrix.png"
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix", width=500)
    else:
        st.info("Confusion matrix not available. Run `python ml/train.py` first.")

    st.divider()

    st.markdown(
        "We trained a binary classifier on the EMSCAD dataset. "
        "The model converts job text into numerical features using TF-IDF, "
        "then classifies each listing as real or fake. We optimize for recall "
        "on fake listings — meaning we prefer to flag a real job as suspicious "
        "rather than miss an actual fake."
    )
