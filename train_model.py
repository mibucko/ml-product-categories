import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
import pickle

# Data preparation
df = pd.read_csv("products.csv")
df.columns = (df.columns.str.strip().str.lower().str.replace(r"[\s_]+", "_", regex=True).str.strip("_"))
df = df.drop(columns=["product_id", "merchant_id", "product_code", "number_of_views",
    "merchant_rating", "listing_date"])
df["category_label"] = df["category_label"].replace({
    "Mobile Phone": "Mobile Phones",
    "CPU": "CPUs",
    "fridge": "Fridges"})
df = df.dropna(subset=["product_title", "category_label"])
df["special_count"] = df["product_title"].str.count(r"[^A-Za-z0-9\s]")

# Separate features and labels
X_text = df["product_title"]
X_num = df[["special_count"]]
y = df["category_label"]

# Vectorizing text data
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    lowercase=True)
X_tfidf = vectorizer.fit_transform(X_text)

#Scaling numerical data
scaler = MinMaxScaler()
X_num_scaled = scaler.fit_transform(X_num)
X_final = hstack([
    X_tfidf,
    X_num_scaled
])

# Model training and saving
model = LinearSVC()
model.fit(X_final, y)

with open("category_predict_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "vectorizer": vectorizer,
        "scaler": scaler
    }, f)

print("\nModel, vectorizer, scaler saved.")
