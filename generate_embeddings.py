import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

# Load your dataset
df = pd.read_csv(r"C:\Users\sanju\Downloads\product-recommendation-system-BERT-main\product-recommendation-system-BERT-main\data\flipkart_com-ecommerce_sample.csv")

# Use the 'product_name' column for generating embeddings
if 'product_name' not in df.columns:
    raise ValueError("❌ 'product_name' column not found in CSV.")

# Remove missing and duplicate product names
df = df[['product_name']].dropna().drop_duplicates().reset_index(drop=True)
df.columns = ['text']

# Check if data is empty
if df.empty:
    raise ValueError("❌ No valid product names found. Check your dataset.")

print(f"✅ Loaded {len(df)} unique product names.")

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Generate CLS embeddings
embeddings = []

with torch.no_grad():
    for text in tqdm(df['text'], desc="Generating BERT embeddings"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # [CLS] token
        embeddings.append(cls_embedding)

# Convert to NumPy array
embeddings = np.array(embeddings)

# Save outputs
np.save("product_embeddings.npy", embeddings)
df.to_csv("product_info.csv", index=False)

print("✅ product_embeddings.npy and product_info.csv saved successfully.")
