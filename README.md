Product-Recommendation-Agent
This project features a smart Product Recommendation AI Agent created using Streamlit, OpenAI GPT, and FAISS embeddings. Users can look for products by typing in questions in everyday language, using filters, and making choices interactively.
AI-Based Product Recommendation System

A Streamlit-powered intelligent recommender application using embeddings and similarity search

📌 1. Overview

This project is an AI-powered Product Recommendation System built using:

Streamlit for an interactive dashboard

OpenAI Embeddings for generating vector representations

Cosine similarity for product match ranking

Pandas / NumPy for data processing

The system takes a user query (e.g., “running shoes for women”) and recommends the most relevant products from a large dataset.

🚀 2. Features

✔ Search instantly using natural language
✔ High-quality product recommendations
✔ Embedding-based similarity search
✔ Clean and modern Streamlit UI
✔ Option to display product images
✔ Works with large CSV datasets
✔ Fast response time using vectorized NumPy operations

🧠 3. Tech Stack
Category	Tools Used
Frontend	Streamlit
Backend	Python
AI Model	OpenAI Embeddings
ML Ops	NumPy, Pandas
Deployment	Local or Cloud
Version Control	Git & GitHub
🏗 4. System Architecture
User Input → Generate Embedding → Compare with Product Embeddings →
Cosine Similarity → Top N Recommendations → Display on UI

Architecture Diagram
                          ┌────────────────────────────┐
                          │        User / Client        │
                          │ (Streamlit UI + Chat Input) │
                          └──────────────┬─────────────┘
                                         │
                                         ▼
                           ┌────────────────────────┐
                           │     Recommendation     │
                           │        Engine          │
                           └────────────┬──────────┘
                                        │
     ┌──────────────────────────┬────────┴──────────┬──────────────────────────┐
     │                          │                   │                          │
     ▼                          ▼                   ▼                          ▼

┌──────────────┐      ┌────────────────┐    ┌────────────────┐      ┌───────────────────────┐
│  Query       │      │ Embedding      │    │ Vector Index   │      │ Product Dataset        │
│ Preprocessor │      │ Model (MiniLM) │    │ (FAISS)        │      │ (CSV / JSON)           │
│  - Clean     │      │  - Sentence    │    │  - Cosine sim  │      │  - Product title       │
│  - Expand    │      │    Embeddings  │    │  - Top-k match │      │  - Description         │
│  - Enrich    │      └────────────────┘    └────────────────┘      │  - Price/category      │
└──────────────┘                                                      └───────────────────────┘
                                         │
                                         ▼
                          ┌────────────────────────────┐
                          │ Large Language Model (LLM) │
                          │ (OpenAI / GPT / Local LLM) │
                          │   - Filters results        │
                          │   - Generates explanation  │
                          └──────────────┬─────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ Final Recommendations│
                              │  - Ranking           │
                              │  - Reasoning         │
                              │  - Product details   │
                              └─────────────────────┘
5. Dataset Details

You are using the file:

products_large.csv


It typically contains:

Column	Description
product_id	Unique ID
name	Product title
description	Product details
category	Product category
price	Product price
image_url	Link to product image

The dataset is converted into embeddings using OpenAI and stored in:

embeddings.npy
6. Recommendation Workflow
Step 1 — Product Dataset

Load CSV file and extract text fields (name + description).

Step 2 — Create Embeddings

Embed each product using OpenAI API and store in embeddings.npy.

Step 3 — User Query

User enters a natural-language search.

Step 4 — Query Embedding

Convert user query into embedding.

Step 5 — Cosine Similarity

Compute similarity:

similarity = dot(query, product_vector) / (||query|| * ||vector||)

Step 6 — Sort + Top N

Return top 5 or 10 most similar products.

Step 7 — Display in Streamlit

Show product name, price, and image.

🖥 7. Screenshots

images/homepage.png
images/recommendations.png

8. Folder Structure
project_recommender/
│── app.py
│── README.md
│── requirements.txt
│── products_large.csv
│── embeddings.npy
│── .gitignore
│── images/                # Add screenshots here
│── .streamlit/
│      └── secrets.toml    # (NOT committed to GitHub)

⚙ 9. Installation & Running Instructions
🔹 Step 1 — Clone the Repository
git clone https://github.com/yourusername/Product-Recommendation-Agent.git
cd Product-Recommendation-Agent

🔹 Step 2 — Create a Virtual Environment
python -m venv venv
venv\Scripts\activate     # Windows

🔹 Step 3 — Install Dependencies
pip install -r requirements.txt

🔹 Step 4 — Add Secrets (API Keys)

Create:

.streamlit/secrets.toml


Inside add:

OPENAI_API_KEY = "your_api_key_here"

🔹 Step 5 — Run Streamlit App
streamlit run app.py


Your system will open in browser at:

http://localhost:8501

🧪 10. Example Output

User query:

"Bluetooth headphones for workouts"


Returned recommendations:

JBL Endurance Active Wireless

Sony WF-1000XM3

Bose Sport Wireless Earbuds

🚀 11. Future Enhancements

Add user authentication

Add product category filtering

Add database backend (PostgreSQL / Firebase)

Add LLM-powered explanation: “Why this product was recommended?”

Deploy to Streamlit Cloud / Render

Add caching to reduce API cost

🤝 12. Contributing

Pull requests are welcome!
Please open an issue if you want to suggest a feature or report a bug.

13.🔐 Adding the OpenAI API Key

This project requires an OpenAI API key to generate embeddings.
You must store your key securely using Streamlit’s secrets system.

Create a folder (if not already present):

.streamlit/


Inside it, create the file:

.streamlit/secrets.toml


Add your API key:

OPENAI_API_KEY = "your_api_key_here"


⚠ Important:
.streamlit/secrets.toml is included in .gitignore and must never be pushed to GitHub for security reasons.

📞 14. Contact

Developer: Rakshitha S M
GitHub: https://github.com/RakshithaMunegowda
Email:rakshithagowdasm62@gmail.com# Product-Recommendation-Agent
This project features a smart Product Recommendation AI Agent created using Streamlit, OpenAI GPT, and FAISS embeddings. Users can look for products by typing in questions in everyday language, using filters, and making choices interactively. 
