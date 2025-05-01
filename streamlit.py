# from graphviz import Digraph
import streamlit as st
import pandas as pd
import numpy as np
import torch
import time
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
import Levenshtein
import plotly.express as px
import mlflow.tracking
from google.cloud import storage
from io import BytesIO
import os

# Set page config for wider layout
st.set_page_config(layout="wide")
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\sanju\AppData\Roaming\gcloud\application_default_credentials.json"
# client = storage.Client()
# bucket = client.get_bucket("mlops_bucket_pr")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/flipkart_com-ecommerce_sample.csv")
    df = df[[
        "uniq_id", "product_name", "description", "brand",
        "product_category_tree", "product_url", "image",
        "discounted_price", "product_rating", "product_specifications"
    ]]
    df = df.dropna(subset=["product_name", "description"])
    df["text"] = df["product_name"] + " " + df["description"]

    import ast
    df["image"] = df["image"].apply(lambda x: ast.literal_eval(x)[0] if pd.notnull(x) and "[" in x else None)
    return df

    # client = storage.Client()
    # bucket = client.get_bucket("mlops_bucket_pr")
    # blob = bucket.blob("flipkart_com-ecommerce_sample.csv")
    # data = blob.download_as_bytes()
    #
    # # Read into pandas
    # df = pd.read_csv(BytesIO(data))
    # df = df[[
    #     "uniq_id", "product_name", "description", "brand",
    #     "product_category_tree", "product_url", "image",
    #     "discounted_price", "product_rating", "product_specifications"
    # ]]
    # df = df.dropna(subset=["product_name", "description"])
    # df["text"] = df["product_name"] + " " + df["description"]
    #
    # import ast
    # df["image"] = df["image"].apply(lambda x: ast.literal_eval(x)[0] if pd.notnull(x) and "[" in x else None)
    # return df


df = load_data()


@st.cache_resource
def load_embeddings():
    return np.load("embeddings5_new.npy")

    # bucket = client.get_bucket("mlops_bucket_pr")
    # blob = bucket.blob("embeddings5_new.npy")
    # data = blob.download_as_bytes()
    #
    # return np.load(BytesIO(data))


@st.cache_resource
def load_model():
    return SentenceTransformer("bert-base-nli-mean-tokens")


product_embeddings = torch.tensor(load_embeddings())
model = load_model()

# Dropdown for page navigation
st.sidebar.title("📂 Navigation")
page = st.sidebar.selectbox("Go to", ["Overview", "Methodology", "Recommendations", "Logs and Analysis"])


def calculate_string_similarities(query, product_name):
    return {
        "fuzzy_ratio": fuzz.ratio(query.lower(), product_name.lower()),
        "fuzzy_partial": fuzz.partial_ratio(query.lower(), product_name.lower()),
        "fuzzy_token": fuzz.token_sort_ratio(query.lower(), product_name.lower()),
        "levenshtein": Levenshtein.ratio(query.lower(), product_name.lower())
    }


def log_activity(query, top_product, top_score, query_time, similarities, df, top_n=10):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "query": query,
        "top_product_name": top_product['product_name'],
        "top_similarity": float(top_score),
        "time_taken": float(query_time),
        "num_recommendations": int(top_n),
        "string_similarities": calculate_string_similarities(query, top_product['product_name']),
        "products": []
    }

    for idx in range(min(top_n, len(similarities))):
        product = df.iloc[idx]
        product_details = {
            "product_name": product['product_name'],
            "similarity": float(similarities[idx].item()),
            "rating": product['product_rating'] if pd.notna(product['product_rating']) else None,
            "string_similarities": calculate_string_similarities(query, product['product_name'])
        }
        log_entry["products"].append(product_details)

    try:
        with open("activity_log.json", "r") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(log_entry)

    with open("activity_log.json", "w") as f:
        json.dump(logs, f, indent=4)


if page == "Overview":
    st.image("103325de784f53c0.png", width=1000)
    st.title("🛍️ SmartShop Recommender")
    st.markdown("Get product recommendations using AI that understands your intent.")
    st.markdown("""
    ### Key Features:
    - Semantic product search
    - Personalized recommendations
    - Performance analytics
    - Interactive visualizations
    """)

elif page == "Methodology":
    st.title("🧠 Methodology")
    if st.button("Click to View Architecture"):
        st.image("mlops-2025-03-20-021715.png", width=1000)
    st.title("How It Works")

    # SECTION 1: Data Preparation
    with st.expander("1️⃣ Data Preparation: Clean & Combine"):
        st.markdown("""
        - Clean raw product data (titles, descriptions, etc.)
        - Merge title + description → one enriched input for embedding

        **Example:**
        ```
        "Redmi Note 11 - 6GB RAM | Powerful camera | Long battery life"
        ```

        This helps us better understand what each product offers 🧐
        """)

    # SECTION 2: Embedding Generation
    with st.expander("2️⃣ Embedding Generation with BERT"):
        st.markdown("""
        - Use `sentence-transformers` to turn product descriptions into high-dimensional vectors
        - Store precomputed embeddings for blazing-fast search ⚡️

        Think of it as giving every product a digital fingerprint 🧬
        """)
        st.image("https://miro.medium.com/v2/resize:fit:1200/1*5HqeXoA_EZyyDsp4Kf7X3Q.gif",
                 caption="Text → Embedding using BERT", use_container_width=True)

    # SECTION 3: Similarity Search
    with st.expander("3️⃣ Find Similar Products with Cosine Similarity"):
        st.markdown("""
        - Convert user query into an embedding using the same BERT model
        - Measure cosine similarity with all product vectors
        - Return top N closest products 🎯

        This is how we match your vibe with the right items!
        """)

    # SECTION 4: Analytics and Monitoring
    with st.expander("4️⃣ ML Analytics & Monitoring (MLflow)"):
        st.markdown("""
        - Log each search query and result using MLflow
        - Track model parameters, metrics, and visualizations
        - Monitor system health and performance over time 📈

        """)

        # # Replace with your actual MLflow run URL dynamically if possible
        # mlflow_url = "http://localhost:5000/#/experiments/0"
        # st.markdown(f"[🔗 View Latest MLflow Run]({mlflow_url})", unsafe_allow_html=True)

    # st.subheader("🔄 End-to-End Pipeline")
    # dot = Digraph()
    # dot.attr(rankdir="LR", size='8,5')
    # dot.attr('node', shape='box', style='filled', fillcolor='#e0f7fa')

    # nodes = [
    #     ("Data", "📁 Data Preparation"),
    #     ("Embed", "🤖 BERT Embedding"),
    #     ("Sim", "📈 Cosine Similarity"),
    #     ("Reco", "🎁 Recommendations"),
    #     ("Track", "📊 MLflow Tracking")
    # ]
    #
    # for key, label in nodes:
    #     dot.node(key, label)
    #
    # dot.edges([
    #     ("Data", "Embed"),
    #     ("Embed", "Sim"),
    #     ("Sim", "Reco"),
    #     ("Reco", "Track")
    # ])
    #
    # st.graphviz_chart(dot)

    st.success("Now you know what powers SmartShop under the hood! 💡")

elif page == "Recommendations":
    st.title("🔍 Product Search & Recommendations")
    query = st.text_input("Enter a product you're looking for:", placeholder="e.g., wireless bluetooth headphones")
    top_n = st.slider("How many recommendations do you want?", min_value=1, max_value=20, value=10)

    if query:
        start_time = time.time()
        with st.spinner("Finding similar products..."):
            query_embedding = model.encode(query, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(query_embedding, product_embeddings)[0]
            top_results = torch.topk(similarities, k=top_n)
            end_time = time.time()

            top_score = similarities[top_results.indices[0]].item()
            top_product = df.iloc[top_results.indices[0].item()]
            query_time = round(end_time - start_time, 3)

            # Log the activity
            log_activity(query, top_product, top_score, query_time,
                         similarities[top_results.indices], df, top_n)

            st.markdown("## 🎯 Top Recommendations")
            # Custom CSS for consistent styling
            st.markdown("""
            <style>
                .product-card {
                    border: 1px solid #e0e0e0;
                    border-radius: 12px;
                    padding: 16px;
                    margin-bottom: 20px;
                    transition: transform 0.2s;
                    height: 100%;
                }
                .product-card:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 6px 12px rgba(0,0,0,0.1);
                }
                .image-container {
                    width: 100%;
                    height: 200px;
                    overflow: hidden;
                    border-radius: 8px;
                    margin-bottom: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: #f8f9fa;
                }
                .product-image {
                    object-fit: contain;
                    max-width: 100%;
                    max-height: 100%;
                }
                .similarity-badge {
                    background: #f8f9fa;
                    border-radius: 4px;
                    padding: 8px;
                    font-size: 0.85rem;
                }
            </style>
            """, unsafe_allow_html=True)

            # Display products in responsive grid
            for i in range(0, len(top_results.indices), 3):
                cols = st.columns(3, gap="medium")
                for j in range(3):
                    if i + j >= len(top_results.indices):
                        break

                    idx = top_results.indices[i + j].item()
                    product = df.iloc[idx]
                    similarities_str = calculate_string_similarities(query, product['product_name'])

                    with cols[j]:
                        # Product card container
                        st.markdown(
                            f"""
                            <div class="product-card">
                                <div class="image-container">
                                    <img class="product-image" src="{product['image'] or 'https://via.placeholder.com/200x200?text=No+Image'}">
                                </div>
                                <a href="{product['product_url']}" target="_blank" style="text-decoration: none;">
                                    <h3 style="color: #0068c9; margin-bottom: 8px; font-size: 1.1rem;">{product['product_name']}</h3>
                                </a>
                                <div style="margin-bottom: 12px;">
                                    <div style="color: #6c757d; font-size: 0.9rem; margin-bottom: 4px;">
                                        {product['brand'] or 'No brand'}
                                    </div>
                                    <div style="font-weight: 600; color: #2e4053; margin-bottom: 8px;">
                                        ₹{product['discounted_price']}
                                    </div>
                                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 12px;">
                                        <div class="similarity-badge" title="Cosine Similarity Score">
                                            🔍 <span style="color: #2c3e50;">{similarities[idx].item():.3f}</span>
                                        </div>
                                        <div class="similarity-badge" title="Fuzzy Match Ratio">
                                            🎯 <span style="color: #27ae60;">{similarities_str['fuzzy_ratio']:.0f}</span>
                                        </div>
                                        <div class="similarity-badge" title="Normalized Levenshtein Similarity">
                                            📏 <span style="color: #2980b9;">{similarities_str['levenshtein']:.2f}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            # MLflow logging
            # mlflow.set_tracking_uri("gs://mlops_bucket_pr/mlflow-logs")
            # mlflow.set_tracking_uri("http://localhost:5000")

            if mlflow.active_run():
                mlflow.end_run()

            with mlflow.start_run():
                mlflow.log_param("query", query)
                mlflow.log_param("top_product", top_product['product_name'])
                mlflow.log_param("num_recommendations", top_n)

                # 📏 Log similarity metrics
                mlflow.log_metric("top_similarity", top_score)
                mlflow.log_metric("query_time", query_time)

                # 🧠 String similarity metrics
                sim_metrics = calculate_string_similarities(query, top_product['product_name'])
                mlflow.log_metric("fuzzy_ratio", sim_metrics['fuzzy_ratio'])
                mlflow.log_metric("levenshtein", sim_metrics['levenshtein'])

            # mlflow.set_tracking_uri("http://localhost:5000")  # or "http://your-cloud-run-url"

            st.markdown("## ⏱ Query Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("Time Taken", f"{query_time} seconds")
            col2.metric("Top Similarity", f"{top_score:.3f}")
            col3.metric("Recommendations", top_n)

elif page == "Logs and Analysis":
    st.title("📊 Logs and Analysis")

    st.subheader("MLflow History")
    client = mlflow.tracking.MlflowClient()

    latest_run = client.search_runs(experiment_ids=["0"], order_by=["start_time DESC"], max_results=1)[0]
    # mlflow_url = f"http://localhost:5000/#/experiments/0/runs/{latest_run.info.run_id}"
    try:
        client = mlflow.tracking.MlflowClient()
        latest_run = client.search_runs(
            experiment_ids=["0"],
            order_by=["start_time DESC"],
            max_results=1
        )[0]
        mlflow_url = f"{MLFLOW_URI}/#/experiments/0/runs/{latest_run.info.run_id}"
    except Exception as e:
        st.error(f"MLflow connection error: {str(e)}")
        mlflow_url = None

    st.link_button("MLFlow UI", mlflow_url)

    try:
        with open("activity_log.json", "r") as f:
            logs = json.load(f)

        if not logs:
            st.warning("No logs found. Perform some searches first.")
        else:
            # Convert logs to DataFrame for analysis
            log_df = pd.DataFrame([{
                'timestamp': log.get('timestamp'),
                'query': log.get('query'),
                'time_taken': log.get('time_taken', 0),
                'top_similarity': log.get('top_similarity', 0),
                'num_recommendations': log.get('num_recommendations', 0),
                'top_product': log.get('top_product_name', ''),
                'fuzzy_ratio': log.get('string_similarities', {}).get('fuzzy_ratio', 0),
                'levenshtein': log.get('string_similarities', {}).get('levenshtein', 0)
            } for log in logs])

            log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
            log_df['hour'] = log_df['timestamp'].dt.hour
            log_df['date'] = log_df['timestamp'].dt.date

            st.subheader("📈 Activity Overview")
            st.write(f"Total searches: {len(log_df)}")

            # Interactive line plot with Plotly
            st.subheader("⏰ Query Timeline")
            fig = px.line(
                log_df.groupby('date').size().reset_index(name='count'),
                x='date',
                y='count',
                title='Daily Search Activity',
                markers=True,
                template='plotly_white'
            )
            fig.update_traces(
                line=dict(width=3, color='#0068c9'),
                marker=dict(size=10, color='#ff4b4b')
            )
            fig.update_layout(
                hovermode="x unified",
                xaxis_title="Date",
                yaxis_title="Number of Searches"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Performance metrics
            st.subheader("⚡ Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Query Time", f"{log_df['time_taken'].mean():.3f} seconds")
            col2.metric("Avg BERT Similarity", f"{log_df['top_similarity'].mean():.3f}")
            col3.metric("Avg Fuzzy Ratio", f"{log_df['fuzzy_ratio'].mean():.0f}")
            col4.metric("Avg Levenshtein", f"{log_df['levenshtein'].mean():.3f}")

            # Most recommended products table
            st.subheader("🏆 Top Recommended Products")
            try:
                # Create a DataFrame of all recommended products
                all_products = []
                for log in logs:
                    if 'products' in log:
                        for product in log['products']:
                            all_products.append({
                                'Product': product['product_name'],
                                'Count': 1,
                                'Avg BERT Score': product['similarity'],
                                'Avg Fuzzy Ratio': product['string_similarities']['fuzzy_ratio'],
                                'Avg Levenshtein': product['string_similarities']['levenshtein']
                            })

                if all_products:
                    products_df = pd.DataFrame(all_products)
                    summary_df = products_df.groupby('Product').agg({
                        'Count': 'count',
                        'Avg BERT Score': 'mean',
                        'Avg Fuzzy Ratio': 'mean',
                        'Avg Levenshtein': 'mean'
                    }).sort_values('Count', ascending=False).head(10).reset_index()

                    # Format the numbers
                    summary_df['Avg BERT Score'] = summary_df['Avg BERT Score'].round(3)
                    summary_df['Avg Fuzzy Ratio'] = summary_df['Avg Fuzzy Ratio'].round(0)
                    summary_df['Avg Levenshtein'] = summary_df['Avg Levenshtein'].round(3)

                    # Display as a table with styling
                    st.dataframe(
                        summary_df.style
                        .background_gradient(cmap='Blues', subset=['Count'])
                        .format({
                            'Avg BERT Score': '{:.3f}',
                            'Avg Fuzzy Ratio': '{:.0f}',
                            'Avg Levenshtein': '{:.3f}'
                        }),
                        use_container_width=True
                    )
                else:
                    st.warning("No product recommendation data available")
            except Exception as e:
                st.error(f"Error processing product data: {str(e)}")

            # Recent activity logs
            st.subheader("📝 Recent Activity Logs")
            for log in logs[-5:]:
                with st.expander(f"{log.get('timestamp', 'No timestamp')} - '{log.get('query', 'No query')}'"):
                    col1, col2 = st.columns(2)
                    col1.metric("Time Taken", f"{log.get('time_taken', 0)}s")
                    col2.metric("Recommendations", log.get('num_recommendations', 0))

                    col1, col2, col3 = st.columns(3)
                    col1.metric("BERT Score", f"{log.get('top_similarity', 0):.3f}")
                    col2.metric("Fuzzy Ratio", f"{log.get('string_similarities', {}).get('fuzzy_ratio', 0):.0f}")
                    col3.metric("Levenshtein", f"{log.get('string_similarities', {}).get('levenshtein', 0):.3f}")

    except FileNotFoundError:
        st.warning("No logs found. Perform some searches first to generate analytics.")
    except json.JSONDecodeError:
        st.error("Error reading log file. The file might be corrupted.")
