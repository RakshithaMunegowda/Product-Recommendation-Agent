import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI

# -------------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------------
st.set_page_config(page_title="AI Shopping Assistant", layout="wide")

# -------------------------------------------------------------
# Load OpenAI API Key
# -------------------------------------------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------------------------------------------
# Load Products dataset
# -------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("products.csv")
    df["text"] = df["title"] + " " + df["description"] + " " + df["category"]
    return df

df = load_data()

# -------------------------------------------------------------
# TF-IDF Encoding for Product Search
# -------------------------------------------------------------
@st.cache_resource
def vectorize_data():
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(df["text"])
    return vectorizer, vectors

vectorizer, vectors = vectorize_data()

# -------------------------------------------------------------
# AI Preference Extractor — safe for Streamlit Cloud
# -------------------------------------------------------------
def extract_preferences(user_query):
    system_prompt = """
    You are a product preference extractor.
    Return this JSON only:
    {
      "category": "...",
      "min_price": number or null,
      "max_price": number or null,
      "brand": "...",
      "features": ["...", "..."]
    }
    If user does not specify something, return null for that field.
    """

    try:
        time.sleep(1)  # prevents OpenAI rate limit
        response = client.chat.completions.create(
            model="gpt-4o-mini-tts",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        return json.loads(response.choices[0].message.content)

    except Exception:
        st.warning("⚠ AI unavailable — fallback to basic keyword matching.")
        return {}

# -------------------------------------------------------------
# Product Recommendation Engine
# -------------------------------------------------------------
def get_recommendations(preferences, top_n=5):
    df_filtered = df.copy()

    if "category" in preferences and preferences["category"]:
        df_filtered = df_filtered[df_filtered["category"].str.contains(preferences["category"], case=False)]

    if "brand" in preferences and preferences["brand"]:
        df_filtered = df_filtered[df_filtered["title"].str.contains(preferences["brand"], case=False)]

    if preferences.get("min_price"):
        df_filtered = df_filtered[df_filtered["price"] >= float(preferences["min_price"])]

    if preferences.get("max_price"):
        df_filtered = df_filtered[df_filtered["price"] <= float(preferences["max_price"])]

    if df_filtered.empty:
        return None

    query_vec = vectorizer.transform([preferences.get("category", "")])
    sims = cosine_similarity(query_vec, vectors[df_filtered.index]).flatten()
    df_filtered["score"] = sims

    return df_filtered.sort_values("score", ascending=False).head(top_n)

# -------------------------------------------------------------
# UI — chatbot + results
# -------------------------------------------------------------
st.title("💬 AI Shopping Assistant")
chat_query = st.text_input("Tell me what you're looking for...")

if chat_query.strip():
    with st.spinner("Understanding your needs..."):
        prefs = extract_preferences(chat_query)
        results = get_recommendations(prefs)

    if results is None:
        st.error("❌ No matching products found. Try again with different keywords.")
    else:
        st.subheader("🛍 Best matches for you:")
        for _, row in results.iterrows():
            with st.container():
                st.markdown(f"### {row['title']}")
                st.write(row['description'])
                st.write(f"💰 **Price:** ₹{row['price']}")
                st.write(f"🏷 **Category:** {row['category']}")
                st.write("---")
