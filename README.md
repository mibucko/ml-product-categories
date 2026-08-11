# ml-product-categories



**Welcome**



You can use the model from this repository to predict a product category by entering the product title.



**How to use it**



a) Open the Python file predict\_category.py and press Run.

b) Enter or paste the product title and press Enter. The predicted category will be displayed.

c) When you want to finish, type exit.



**Description of Model Development**



The model was trained on the dataset products.csv, which can be found in this repository.

All stages of the model development are documented in the Colab notebook product\_categories\_prediction.ipynb.



**First stage:** Column names were standardized, and several columns that were considered unrelated to the product category were removed.

**Second stage:** We examined whether the product\_code column was related to the product category. Based on visual inspection, we concluded that it was not, so this column was also removed.

**Third stage:** Similar categories were merged, for example, fridge and Fridges.



**Preparation for Better Model**

**Fourth stage**: Five new features were created from the product title: word\_count, char\_count, special\_count, words\_with\_digits, and max\_word\_length.

The aim was to examine whether any of these features could improve the model's performance.



**Fifth Stage:** Data Preparation for Algorithm Training

* We created seven experimental datasets (e.g., 1. Baseline, 2. Baseline + word\_count, etc.).
* The data was split into training and test sets.
* The product\_title column was vectorized using TF-IDF. We experimented with unigrams, bigrams, and trigrams.
* The remaining numerical features were scaled.
* For better organization and readability of the subsequent cells, two helper functions were defined: one for dataset selection and another for model training and evaluation.



**Models evaluation**

**Sixth Stage:** We evaluated five different models on seven experimental datasets, resulting in a total of 35 tests.

Based on the evaluation metrics, the Support Vector Machine (SVM) achieved the best overall performance. When comparing the different datasets, the best-performing combination was the baseline dataset (product\_title) supplemented with the special\_count feature.

It should be noted that the improvement was very small, only approximately 0.01 in F1-score. Therefore, if adding this feature causes any complications in future applications, it is reasonable to use the baseline dataset alone.



**Seventh Stage:** Comparison of Vectorization Approaches

We compared three types of TF-IDF vectorization: unigram, bigram, and trigram. The bigram approach showed a very small improvement, again approximately 0.01 in F1-score. Therefore, all three approaches can be considered viable for future use. For the final model, however, we selected bigram vectorization.



As a result of this analysis, we created the category\_predict\_model, which contains the trained model, vectorizer, and scaler in a single file.

Finally, users can use the model by opening the Python file predict\_category.py.





