import pickle
import re
from scipy.sparse import hstack
import pandas as pd

# Model, vectorizer and scaler import
with open("category_predict_model.pkl", "rb") as f:
    data = pickle.load(f)
model = data["model"]
vectorizer = data["vectorizer"]
scaler = data["scaler"]

# Function for input preparation
def prepare_user_input(user_input):
    # Transform user input to TF-IDF
    user_tfidf = vectorizer.transform([user_input])

    # Calculate and scale special character count
    special_count = len(re.findall(r"[^A-Za-z0-9\s]", user_input))
    user_num_scaled = scaler.transform(pd.DataFrame([[special_count]], columns=["special_count"]))

    # Combine TF-IDF and numerical feature
    user_final = hstack([user_tfidf, user_num_scaled ])

    return user_final

# Loop for user input and prediction of category
print("\nEnter a product title to classify its category.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter a title: ").strip()
    if user_input.lower() == 'exit':
        print("Thank you for using this model.")
        break
     
    user_final = prepare_user_input(user_input)

    prediction = model.predict(user_final)[0]

    print(f"Predicted category: {prediction}\n")
