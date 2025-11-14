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
    import requests  # Lazy-load requests

    search_url = "https://serpapi.com/search"

    # Format dates for SERP API - use cd_min/cd_max for google engine
    date_filter = f"cd_min:{from_date.strftime('%m/%d/%Y')},cd_max:{to_date.strftime('%m/%d/%Y')}"

    params = {
        "engine": "google",  # Use 'google' engine
        "q": query,
        "api_key": api_key,
        "num": num_articles,  # Ask for this many, but we'll slice later
        "tbs": f"cdr:1,{date_filter}",  # Custom date range
        "tbm": "nws",  # Search the "News" tab
        "gl": "us",    # Geolocation: United States
        "hl": "en"     # Language: English
    }

    # Optional: bias results toward a specific location
    if location_query:
        params["location"] = location_query

    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"SERP API request failed: {e}")
        return None


def get_source_name(article):
    """
    Safely extracts the source name from an article, handling both
    string and object formats for the 'source' field.
    """
    source = article.get("source")
    if isinstance(source, dict):
        return source.get("name", "Unknown")
    elif isinstance(source, str):
        return source
    return "Unknown"


def summarize_with_gemini(api_key, articles):
    """
    Summarizes a list of articles using the Gemini API.
    """
    import google.generativeai as genai  # Lazy-load genai

    genai.configure(api_key=api_key)

    # Prepare article snippets for the prompt
    prompt_data = [
        {
            "title": article.get("title", "No Title"),
            "snippet": article.get("snippet", "No Snippet"),
            "source": get_source_name(article),
            "original_date": article.get("date", "Unknown")
        }
        for article in articles
    ]

    if not prompt_data:
        st.warning("No articles found to summarize.")
        return []

    system_prompt = """
    You are an expert incident analyst. Your task is to extract specific information 
    from a list of news article snippets.
    
    For EACH article, you MUST extract the following fields:
    1.  `location`: The city, state, or specific address mentioned (e.g., "Houston, TX", "Main St, Springfield"). If no specific location is found, use "Unknown".
    2.  `incident_type`: The type of event (e.g., "House Fire", "Chemical Spill", "Wildfire", "Natural Disaster", "Explosion").
    3.  `incident_date`: The specific date the incident occurred, as mentioned in the text. If not mentioned, use "Unknown".
    4.  `source`: The "source" string provided in the input for that article.
    5.  `summary`: A concise 1-2 sentence summary of the incident described.
    
    Respond ONLY with a valid JSON array, where each object in the array
    corresponds to an article and contains the fields listed above.
    """

    model = genai.GenerativeModel(
        'gemini-2.5-flash-preview-09-2025',
        system_instruction=system_prompt
    )

    user_prompt = f"""
    Here is a list of news articles in JSON format:
    {json.dumps(prompt_data, indent=2)}
    
    Please extract the requested information for each article and return
    a valid JSON array.
    """

    response = None
    try:
        response = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )

        summaries = json.loads(response.text)

        # Attach original article data
        for i, summary in enumerate(summaries):
            if i < len(articles):
                summary['article_date'] = articles[i].get("date", "Unknown")
                summary['article_link'] = articles[i].get("link", "#")
                summary['article_title'] = articles[i].get("title", "No Title")

        return summaries

    except Exception as e:
        st.error(f"Gemini API request failed: {e}")
        raw_text = "No response object was created."
        if response and hasattr(response, 'text'):
            raw_text = response.text
        elif response:
            raw_text = str(response)

        st.error(f"Gemini raw response (if available): {raw_text}")
        return []


def convert_df_to_csv(df):
    """
    Converts a DataFrame to a CSV string.
    """
    return df.to_csv(index=False).encode('utf-8')


# --- Main Streamlit App ---

def main():
    # Page config must be first Streamlit call
    st.set_page_config(
        page_title="Incident Summarizer",
        page_icon="🔥",
        layout="wide"
    )

    # --- Load secrets from Streamlit ---
    try:
        SERP_API_KEY = st.secrets["SERP_API_KEY"]
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        st.error("Missing SERP_API_KEY or GEMINI_API_KEY in Streamlit secrets.")
        st.stop()

    APP_PASSWORD = st.secrets["APP_PASSWORD"] if "APP_PASSWORD" in st.secrets else None

    # --- Password Protection ---
    if APP_PASSWORD:
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False

        if not st.session_state.authenticated:
            st.title("🔐 Protected Incident Reporter")
            pw = st.text_input("Enter password", type="password")

            if pw == APP_PASSWORD:
                st.session_state.authenticated = True
                st.success("Access granted!")
                st.rerun()
            elif pw:
                st.error("Incorrect password.")
                st.stop()
            else:
                st.stop()
    else:
        st.warning("No APP_PASSWORD set in Streamlit secrets — app is not password protected.")

    # --- Sidebar ---
    st.sidebar.title("Search Configuration")
    st.sidebar.success("API keys loaded successfully from Streamlit secrets")

    # Metro area selection (sorted, Nationwide first)
    metros_sorted = ["Nationwide"] + sorted(
        [m for m in TOP_50_US_METROS if m != "Nationwide"]
    )

    metro_area = st.sidebar.selectbox(
        "Select Metro Area (or Nationwide)",
        options=metros_sorted,
        index=0  # Nationwide at top
    )

    # Number of articles
    num_articles = st.sidebar.slider(
        "Number of Articles to Summarize",
        min_value=5,
        max_value=25,
        value=10
    )

    # Date range
    st.sidebar.subheader("Date Range")
    default_end_date = datetime.now()
    default_start_date = default_end_date - timedelta(days=2)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        from_date = st.date_input("From", value=default_start_date)
    with col2:
        to_date = st.date_input("To", value=default_end_date)

    # Validate date range
    if from_date > to_date:
        st.sidebar.error("Error: 'From' date must be before 'To' date.")
        return

    # --- Main Page ---
    st.title("ATI News Incident Reporter")
    st.markdown("Get AI-powered summaries of recent incidents from Google News. Select the number of articles to pull and the MSAs if applicable.")

    if st.button("Search for Incidents", type="primary"):

        # --- 1. Define Search Query ---
        incident_keywords = [
            "fire", "explosion", "natural disaster", "chemical spill", "hazmat",
            "building collapse", "power outage", "gas leak", "train derailment",
            "highway closure", "industrial accident", "wildfire", "house fire",
            "apartment fire", "structure fire", "crash", "evacuation", "shelter-in-place"
        ]

        keyword_query = " OR ".join(f'"{k}"' for k in incident_keywords)

        # Bake metro into the query text
        if metro_area == "Nationwide":
            # Nationwide: just the incident keywords
            search_query = f"({keyword_query})"
            location_param = None
        else:
            # Extract the city part ("Los Angeles" from "Los Angeles, CA")
            city = metro_area.split(",")[0].strip()

            # Require the city name to appear in the article text
            search_query = f"({keyword_query}) \"{city}\""

            # Optional: keep location bias as well
            location_param = metro_area

        # --- 2. Run Search & Summarization ---
        with st.spinner("Searching Google News..."):
            search_results = get_serp_api_results(
                SERP_API_KEY,
                search_query,
                100,  # Request up to 100; we'll slice later
                from_date,
                to_date,
                location_query=location_param
            )

        if search_results and "news_results" in search_results:
            all_articles = search_results["news_results"]

            # Respect the slider
            articles_to_summarize = all_articles[:num_articles]

            st.success(
                f"Found {len(all_articles)} articles. "
                f"Summarizing the top {len(articles_to_summarize)} with Gemini..."
            )

            with st.spinner("AI is reading and summarizing..."):
                summaries = summarize_with_gemini(GEMINI_API_KEY, articles_to_summarize)

            if summaries:
                import pandas as pd

                st.subheader("Incident Summaries")

                try:
                    df = pd.DataFrame(summaries)

                    # Reorder columns for readability
                    all_cols = [
                        'article_date', 'incident_date', 'incident_type',
                        'location', 'summary', 'source', 'article_title', 'article_link'
                    ]
                    display_cols = [col for col in all_cols if col in df.columns]
                    df_display = df[display_cols]

                    st.dataframe(df_display)

                    # Download
                    csv_data = convert_df_to_csv(df_display)
                    st.download_button(
                        label="Download Data as CSV",
                        data=csv_data,
                        file_name=f"incident_report_{metro_area}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                    )

                except Exception as e:
                    st.error(f"Failed to create DataFrame or display results: {e}")
                    st.json(summaries)

            else:
                st.warning("Failed to generate summaries from Gemini.")

        elif search_results:
            st.warning("No 'news_results' found in API response. Full response:")
            st.json(search_results)
        else:
            st.error("No search results returned from SERP API.")


if __name__ == "__main__":
    main()

