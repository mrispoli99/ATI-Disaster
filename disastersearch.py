import streamlit as st
import json
from datetime import datetime, timedelta

# --- Constants ---

TOP_50_US_METROS = [
    "Nationwide",
    "New York, NY",
    "Los Angeles, CA",
    "Chicago, IL",
    "Dallas-Fort Worth, TX",
    "Houston, TX",
    "Washington, DC",
    "Miami, FL",
    "Philadelphia, PA",
    "Atlanta, GA",
    "Phoenix, AZ",
    "Boston, MA",
    "San Francisco, CA",
    "Riverside-San Bernardino, CA",
    "Detroit, MI",
    "Seattle, WA",
    "Minneapolis-St. Paul, MN",
    "San Diego, CA",
    "Tampa, FL",
    "Denver, CO",
    "St. Louis, MO",
    "Baltimore, MD",
    "Charlotte, NC",
    "Orlando, FL",
    "San Antonio, TX",
    "Portland, OR",
    "Sacramento, CA",
    "Austin, TX",
    "Pittsburgh, PA",
    "Las Vegas, NV",
    "Cincinnati, OH",
    "Kansas City, MO",
    "Columbus, OH",
    "Indianapolis, IN",
    "Cleveland, OH",
    "San Jose, CA",
    "Nashville, TN",
    "Virginia Beach, VA",
    "Providence, RI",
    "Milwaukee, WI",
    "Jacksonville, FL",
    "Oklahoma City, OK",
    "Raleigh, NC",
    "Memphis, TN",
    "Richmond, VA",
    "New Orleans, LA",
    "Louisville, KY",
    "Salt Lake City, UT",
    "Hartford, CT",
    "Buffalo, NY",
    "Birmingham, AL"
]


# --- Helper Functions ---

def get_serp_api_results(api_key, query, num_articles, from_date, to_date, location_query=None):
    """
    Queries the SERP API (Google News) for the given query.
    """
    import requests

    search_url = "https://serpapi.com/search"

    date_filter = f"cd_min:{from_date.strftime('%m/%d/%Y')},cd_max:{to_date.strftime('%m/%d/%Y')}"

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": num_articles,
        "tbs": f"cdr:1,{date_filter}",
        "tbm": "nws",
        "gl": "us",
        "hl": "en"
    }

    if location_query:
        params["location"] = location_query

    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"SERP API request failed: {e}")
        return None


def get_source_name(article):
    source = article.get("source")
    if isinstance(source, dict):
        return source.get("name", "Unknown")
    elif isinstance(source, str):
        return source
    return "Unknown"


def summarize_with_gemini(api_key, articles):
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    prompt_data = [
        {
            "title": a.get("title", "No Title"),
            "snippet": a.get("snippet", "No Snippet"),
            "source": get_source_name(a),
            "original_date": a.get("date", "Unknown")
        } for a in articles
    ]

    if not prompt_data:
        st.warning("No articles found to summarize.")
        return []

    system_prompt = """
    You are an expert incident analyst. Your task is to extract specific information 
    from a list of news article snippets.
    For EACH article, extract:
    1. location
    2. incident_type
    3. incident_date
    4. source
    5. summary
    Return ONLY a valid JSON array.
    """

    model = genai.GenerativeModel(
        "gemini-2.5-flash-preview-09-2025",
        system_instruction=system_prompt
    )

    user_prompt = f"""
    Here is JSON of articles:
    {json.dumps(prompt_data, indent=2)}
    Extract required fields and return JSON.
    """

    try:
        resp = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        summaries = json.loads(resp.text)

        # add metadata
        for i, s in enumerate(summaries):
            s["article_date"] = articles[i].get("date", "Unknown")
            s["article_link"] = articles[i].get("link", "#")
            s["article_title"] = articles[i].get("title", "No Title")

        return summaries

    except Exception as e:
        st.error(f"Gemini summarization failed: {e}")
        return []


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


# --- APP START ---

def main():

    st.set_page_config(
        page_title="Incident Summarizer",
        page_icon="🔥",
        layout="wide"
    )

    # ---- LOAD SECRETS (Streamlit Cloud) ----
    try:
        SERP_API_KEY = st.secrets["SERP_API_KEY"]
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("Missing SERP_API_KEY or GEMINI_API_KEY in Streamlit secrets.")
        st.stop()

    APP_PASSWORD = st.secrets.get("APP_PASSWORD", None)

    # ---- PASSWORD GATE ----
    if APP_PASSWORD:
        if "authed" not in st.session_state:
            st.session_state.authed = False

        if not st.session_state.authed:
            st.title("🔐 Protected Incident Reporter")

            pw = st.text_input("Enter password:", type="password")

            if pw == APP_PASSWORD:
                st.session_state.authed = True
                st.success("Access granted!")
                st.rerun()
            elif pw:
                st.error("Incorrect password.")
                st.stop()
            else:
                st.stop()

    # ---- SIDEBAR ----
    st.sidebar.title("Search Options")
    st.sidebar.success("Secrets loaded successfully")

    metros_sorted = ["Nationwide"] + sorted(
        [m for m in TOP_50_US_METROS if m != "Nationwide"]
    )

    metro_area = st.sidebar.selectbox(
        "Metro Area",
        options=metros_sorted
    )

    num_articles = st.sidebar.slider("Articles to Summarize", 5, 25, 10)

    default_end = datetime.now()
    default_start = default_end - timedelta(days=2)

    col1, col2 = st.sidebar.columns(2)
    from_date = col1.date_input("From", value=default_start)
    to_date = col2.date_input("To", value=default_end)

    if from_date > to_date:
        st.sidebar.error("Invalid date range.")
        return

    # ---- MAIN UI ----
    st.title("🔥 Local & National Incident Reporter")
    st.write("Summaries powered by Google News + Gemini 2.5 Flash.")

    if st.button("Search for Incidents", type="primary"):

        # build query
        keywords = [
            "fire", "explosion", "natural disaster", "chemical spill", "hazmat",
            "building collapse", "power outage", "gas leak", "train derailment",
            "highway closure", "industrial accident", "wildfire", "house fire",
            "apartment fire", "structure fire", "crash", "evacuation", "shelter-in-place"
        ]
        keyword_query = " OR ".join(f'"{k}"' for k in keywords)

        if metro_area == "Nationwide":
            search_query = f"({keyword_query})"
            location_param = None
        else:
            city = metro_area.split(",")[0].strip()
            search_query = f"({keyword_query}) \"{city}\""
            location_param = metro_area

        with st.spinner("Searching Google News..."):
            results = get_serp_api_results(
                SERP_API_KEY,
                search_query,
                100,
                from_date,
                to_date,
                location_query=location_param
            )

        if results and "news_results" in results:
            articles = results["news_results"][:num_articles]

            st.success(f"Found {len(results['news_results'])} articles. Summarizing {len(articles)}...")

            with st.spinner("Summarizing with Gemini..."):
                summaries = summarize_with_gemini(GEMINI_API_KEY, articles)

            if summaries:
                import pandas as pd
                df = pd.DataFrame(summaries)

                order = [
                    "article_date", "incident_date", "incident_type",
                    "location", "summary", "source",
                    "article_title", "article_link"
                ]
                df = df[[c for c in order if c in df.columns]]

                st.dataframe(df)

                csv = convert_df_to_csv(df)
                st.download_button(
                    "Download CSV",
                    csv,
                    file_name=f"incident_report_{metro_area}_{datetime.now().strftime('%Y%m%d')}.csv"
                )
            else:
                st.warning("Gemini returned no summaries.")

        else:
            st.error("No news results found.")


if __name__ == "__main__":
    main()
