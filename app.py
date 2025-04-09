import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import BertTokenizer
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(page_title="Flipkart Product Recommendation", layout="wide")

# === Load Flipkart Data ===
DATA_PATH = r"C:\Users\sanju\Downloads\product-recommendation-system-BERT-main\product-recommendation-system-BERT-main\data\flipkart_com-ecommerce_sample.csv"  # Change if needed

@st.cache_data
def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        st.error(f"File not found at {path}")
        return pd.DataFrame()

df = load_data(DATA_PATH)

# === Sidebar ===
st.sidebar.title("📊 Steps in Data Pipeline")
step = st.sidebar.radio(
    "Navigate:",
    [
        "📁 Load Flipkart Data",
        "🔍 Text Preprocessing",
        "🧠 Tokenization",
        "📊 Visual Insights",
        "🧪 Train-Test Split",
        "⚙️ Model Training Overview",
        "🎯 Try Recommendation"
    ]
)

st.title("🛍️ Flipkart Product Recommendation System using BERT")

# === STEP 1: Load Data ===
if step == "📁 Load Flipkart Data":
    st.subheader("📁 Flipkart Dataset")
    if not df.empty:
        st.success("Data loaded successfully!")
        st.dataframe(df.head(10), use_container_width=True)

        with st.expander("📌 Dataset Info"):
            st.write(df.info())
            st.write(df.describe())

    else:
        st.warning("Please place `flipkart_data.csv` in this folder.")

# === STEP 2: Text Preprocessing ===
elif step == "🔍 Text Preprocessing":
    st.subheader("🔍 Sample Text Preprocessing")
    if not df.empty:
        text_col = st.selectbox("Select text column", df.columns)
        st.write("Original Text Sample:")
        st.text(df[text_col].dropna().iloc[0])

        st.markdown("**After Lowercasing & Punctuation Removal:**")
        import re
        processed_text = re.sub(r'[^a-zA-Z\s]', '', df[text_col].dropna().iloc[0].lower())
        st.code(processed_text)
    else:
        st.warning("Data not loaded.")

# === STEP 3: Tokenization ===
elif step == "🧠 Tokenization":
    st.subheader("🧠 BERT Tokenization Preview")
    if not df.empty:
        sample_text = df.select_dtypes(include='object').dropna().iloc[0, 0]
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer.tokenize(sample_text)
        st.write("Sample Text:", sample_text)
        st.write("Tokenized Output:", tokens[:20])  # Preview first 20
    else:
        st.warning("Data not loaded.")

# === STEP 4: Visual Insights ===
elif step == "📊 Visual Insights":
    st.subheader("📊 Visual Exploration")

    if not df.empty:
        col1, col2 = st.columns(2)

        with col1:
            cat_col = st.selectbox("Select a categorical column for bar chart", df.select_dtypes(include='object').columns)
            st.write(f"Top Categories in {cat_col}")
            top_cats = df[cat_col].value_counts().nlargest(10)
            fig, ax = plt.subplots()
            sns.barplot(x=top_cats.values, y=top_cats.index, ax=ax)
            st.pyplot(fig)

        with col2:
            text_col = st.selectbox("Select a column for WordCloud", df.select_dtypes(include='object').columns, key="text_wc")
            st.write("Word Cloud:")
            text = " ".join(df[text_col].dropna().astype(str).values)
            wordcloud = WordCloud(background_color='white', width=400, height=300).generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

# === STEP 5: Train-Test Split ===
elif step == "🧪 Train-Test Split":
    st.subheader("🧪 Dataset Split Overview")
    st.markdown("Typically 80% of data is used for training and 20% for testing.")
    st.code("""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    """)

# === STEP 6: Model Training Overview ===
elif step == "⚙️ Model Training Overview":
    st.subheader("⚙️ Training with BERT Embeddings")
    st.markdown("""
    The BERT model generates vector embeddings for product text, which are passed through a dense layer
    for classification or similarity-based recommendation.

    **Architecture:**  
    - Pretrained BERT → Embedding  
    - Dense Layer(s) → Output Label (Recommended Category/Product)  
    - Loss: Cross-Entropy or Triplet Loss  
    - Optimizer: AdamW
    """)

# === STEP 7: Try Recommendation ===
# elif step == "🎯 Try Recommendation":
#     st.subheader("🎯 Try Out Product Recommendation")
#
#     user_input = st.text_area("Enter a product description or keywords:")
#     if user_input:
#         st.write("🔍 Tokenizing input with BERT...")
#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         tokens = tokenizer.tokenize(user_input)
#         st.write("📦 Token Preview:", tokens[:10])
#
#         # Mock output
#         st.success("✅ Predicted Recommendation:")
#         st.markdown("**Category:** Electronics \n**Recommended Product:** 'boAt Airdopes 441 Bluetooth Earbuds'")
#
#     else:
#         st.info("Enter some text to get a recommendation.")

elif step == "🎯 Try Recommendation":


    # Load the Flipkart dataset
    # df = pd.read_csv("flipkart_data.csv")
    df = load_data(DATA_PATH)

    # Clean and extract product title and category
    df['product_name'] = df['product_name'].astype(str)

    # Some datasets have category as a list in a string. Clean that.
    if 'product_category_tree' in df.columns:
        df['category'] = df['product_category_tree'].str.extract(r"\[\['(.*?)'")  # Get first category if nested
    else:
        df['category'] = "Unknown"

    # Select only necessary columns
    product_info_df = df[['product_name', 'category']].dropna()

    # Remove duplicates
    product_info_df = product_info_df.drop_duplicates().reset_index(drop=True)

    # Rename columns for consistency
    product_info_df.columns = ['title', 'category']

    # Save to CSV
    product_info_df.to_csv("product_info.csv", index=False)

    print("✅ Saved as product_info.csv")

    st.subheader("🎯 Try Out Product Recommendation")

    user_input = st.text_area("Enter a product description or keywords:")


    @st.cache_resource
    def load_model_and_data():
        # Load product info
        info_df = pd.read_csv(r"C:\Users\sanju\Downloads\product-recommendation-system-BERT-main\product-recommendation-system-BERT-main\product_info.csv")  # Must have 'title', 'category' columns
        # Load pre-computed embeddings
        embeddings = np.load(r"C:\Users\sanju\Downloads\product-recommendation-system-BERT-main\product-recommendation-system-BERT-main\product_embeddings.npy")  # Shape: (num_products, embedding_dim)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return info_df, embeddings, tokenizer


    info_df, product_embeddings, tokenizer = load_model_and_data()


    @st.cache_resource
    def load_bert_model():
        from transformers import BertModel
        import torch
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
        return model


    bert_model = load_bert_model()


    def get_bert_embedding(text, tokenizer, model):
        import torch
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()  # Use [CLS] token
        return cls_embedding


    if user_input:
        st.write("🔍 Generating BERT embedding...")
        user_embedding = get_bert_embedding(user_input, tokenizer, bert_model)  # Shape: (1, 768)

        # Compute similarity
        similarities = cosine_similarity(user_embedding, product_embeddings)  # Shape: (1, num_products)
        top_idx = np.argmax(similarities)

        recommended_title = info_df.iloc[top_idx]['title']
        recommended_category = info_df.iloc[top_idx]['category']

        st.success("✅ Predicted Recommendation:")
        st.markdown(f"**Category:** {recommended_category}")
        st.markdown(f"**Recommended Product:** {recommended_title}")

    else:
        st.info("Enter some text to get a recommendation.")

