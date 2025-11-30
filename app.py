import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import ast
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss

# ----------------------
# Streamlit config
# ----------------------
st.set_page_config(page_title="AI Product Recommender", layout="wide")

# ----------------------
# Paths & constants
# ----------------------
ROOT = Path(__file__).parent
DATA_FILE = ROOT / "products_large.csv"

# ----------------------
# Helpers
# ----------------------
def safe_literal_eval(x):
    if pd.isna(x):
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return list(val)
        return [str(val)]
    except Exception:
        if ">>" in s:
            return [p.strip() for p in s.split(">>") if p.strip()] or [s]
        if "|" in s:
            return [p.strip() for p in s.split("|") if p.strip()] or [s]
        return [s]

def extract_first_image_val(x):
    arr = safe_literal_eval(x)
    return str(arr[0]) if arr else ""

def extract_first_category(x):
    arr = safe_literal_eval(x)
    return str(arr[0]) if arr else "Other"

def map_columns(df):
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df.columns = [c.lower().strip() for c in df.columns]

    mapping = {}
    for cand in ("product_name", "name", "title"):
        if cand in df.columns:
            mapping[cand] = "name"
            break
    for cand in ("description", "product_description", "desc"):
        if cand in df.columns:
            mapping[cand] = "description"
            break
    for cand in ("product_category_tree", "category", "product_category"):
        if cand in df.columns:
            mapping[cand] = "category"
            break
    for cand in ("discounted_price", "retail_price", "price", "cost", "mrp", "selling_price"):
        if cand in df.columns:
            mapping[cand] = "price"
            break
    for cand in ("image", "image_url", "img", "images"):
        if cand in df.columns:
            mapping[cand] = "image_url"
            break
    for cand in ("brand",):
        if cand in df.columns:
            mapping[cand] = "brand"
            break
    for cand in ("overall_rating", "product_rating", "rating", "stars"):
        if cand in df.columns:
            mapping[cand] = "rating"
            break
    for cand in ("product_specifications", "product_spec", "specifications", "features"):
        if cand in df.columns:
            mapping[cand] = "features"
            break

    df = df.rename(columns=mapping)
    return df

# ----------------------
# Load & preprocess dataset
# ----------------------
@st.cache_data
def load_and_prepare(path: Path):
    df = pd.read_csv(path, encoding="utf-8", low_memory=True)
    df = map_columns(df)

    expected_cols = {
        "name": "", "description": "", "category": "Other",
        "brand": "Unknown", "price": 0, "rating": 0.0,
        "image_url": "", "features": ""
    }
    for col, default in expected_cols.items():
        if col not in df.columns:
            df[col] = default

    df["image_url"] = df["image_url"].apply(extract_first_image_val)
    df["category"] = df["category"].apply(extract_first_category)
    df["name"] = df["name"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["brand"] = df["brand"].fillna("Unknown").astype(str)
    df["features"] = df["features"].fillna("").astype(str)

    df["price"] = pd.to_numeric(df["price"].astype(str).str.replace(r"[^\d\.]", "", regex=True),
                                errors="coerce").fillna(0).astype(float)
    df["rating"] = pd.to_numeric(df["rating"].astype(str).str.replace(r"[^\d\.]", "", regex=True),
                                 errors="coerce").fillna(0.0).astype(float)

    df["combined_text"] = (df["name"] + " | " + df["brand"] + " | " + df["features"] + " | " + df["description"]).astype(str)
    df = df.reset_index(drop=True)
    return df

if not DATA_FILE.exists():
    st.error(f"Dataset not found at: {DATA_FILE}")
    st.stop()

df = load_and_prepare(DATA_FILE)

# ----------------------
# TF-IDF vectorization
# ----------------------
@st.cache_data
def build_tfidf(corpus):
    tf = TfidfVectorizer(stop_words="english", max_features=10000)
    mat = tf.fit_transform(corpus)
    return tf, mat

tf, tfidf_matrix = build_tfidf(df["combined_text"].tolist())

# ----------------------
# FAISS embeddings
# ----------------------
@st.cache_resource
def build_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, index, embeddings

emb_model, emb_index, emb_vectors = build_embeddings()

def find_similar_product(product_index, k=6):
    vec = emb_vectors[product_index].reshape(1, -1)
    distances, indices = emb_index.search(vec, k)
    return indices[0][1:]

# ----------------------
# User memory
# ----------------------
if "user_memory" not in st.session_state:
    st.session_state.user_memory = {"brand": {}, "category": {}}

def memory_score(r):
    b = st.session_state.user_memory["brand"].get(r["brand"], 0)
    c = st.session_state.user_memory["category"].get(r["category"], 0)
    return b*2 + c

# ----------------------
# Streamlit UI
# ----------------------
st.title("AI Product Recommender")
st.write("Search and filter products below.")

# Search input
search_query = st.text_input("Search products (name, brand, features, description)")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    unique_cats = ["All"] + sorted(df["category"].dropna().unique().tolist())
    sel_cat = st.selectbox("Category", unique_cats)
    min_price, max_price = float(df["price"].min()), float(df["price"].max())
    pr = st.slider("Price range", min_value=float(min_price), max_value=float(max_price),
                   value=(min_price, max_price))
    sel_brand = st.text_input("Brand (optional)")

# ----------------------
# Apply filters and search
# ----------------------
results = df.copy()

# Category filter
if sel_cat != "All":
    results = results[results["category"] == sel_cat]

# Price filter
results = results[(results["price"] >= pr[0]) & (results["price"] <= pr[1])]

# Brand filter
if sel_brand.strip():
    results = results[results["brand"].str.contains(sel_brand.strip(), case=False, na=False)]

# Search filter
if search_query.strip():
    q_vec = tf.transform([search_query])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    results["search_score"] = results.index.map(lambda i: float(sims[i]) if i < len(sims) else 0.0)
    results = results.sort_values(by="search_score", ascending=False)
else:
    results["memory_boost"] = results.apply(memory_score, axis=1)
    results = results.sort_values(by=["memory_boost", "rating"], ascending=[False, False])

# Top N
TOP_N = st.number_input("Number of results", min_value=1, max_value=50, value=12, step=1)
results = results.head(TOP_N)

# ----------------------
# Grid display
# ----------------------
if results.empty:
    st.info("No products match your query / filters.")
else:
    cols = st.columns(3)
    for idx, (_, row) in enumerate(results.iterrows()):
        col = cols[idx % 3]
        col.markdown(f"**{row['name']}**")
        col.write(f"Brand: {row['brand']}")
        col.write(f"Category: {row['category']}")
        col.write(f"Price: ₹{int(row['price']) if row['price'] else 'N/A'}")
        col.write(f"Rating: ⭐ {row['rating']}")
        try:
            col.image(row["image_url"] if row["image_url"] else "https://via.placeholder.com/200x200.png?text=No+Image", use_container_width=True)
        except Exception:
            col.write("Image not available")
        if col.button("Show similar", key=f"sim_{idx}"):
            item_index = row.name
            sim_idx = find_similar_product(item_index)
            st.subheader(f"Similar to: {row['name']}")
            sim_cols = st.columns(3)
            for j, si in enumerate(sim_idx):
                r = df.iloc[si]
                sc = sim_cols[j % 3]
                sc.markdown(f"**{r['name']}**")
                sc.write(f"₹{int(r['price']) if r['price'] else 'N/A'} — {r['brand']}")
                try:
                    sc.image(r["image_url"] if r["image_url"] else "https://via.placeholder.com/200x200.png?text=No+Image", use_container_width=True)
                except Exception:
                    sc.write("Image not available")

st.markdown("---")
st.write("Dataset loaded from:", str(DATA_FILE.name))
