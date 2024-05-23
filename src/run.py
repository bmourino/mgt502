import os
import time
import nltk
import pandas as pd
from augment import Augmenter  
from embed import BertEmbedder
from nltk.corpus import stopwords
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('../available_datasets/training_data.csv')

# Pipeline with TF-IDF, Scaler, and Logistic Regression
pipeline = Pipeline([
    ('features', FeatureUnion([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('bert', BertEmbedder()),
        ('augmenter', Augmenter())
    ])),
    ('scaler', MaxAbsScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

X = df['sentence'].values
y = df['difficulty'].values

# remove stopwords
stop_words = set(stopwords.words('french'))
X = [' '.join([word for word in sentence.split() if word.lower() not in stop_words]) for sentence in X]

data_unlabelled = pd.read_csv(os.path.abspath("../available_datasets/unlabelled_test_data.csv"))
X_unlabelled = data_unlabelled['sentence'].values
X_unlabelled = [' '.join([word for word in sentence.split() if word.lower() not in stop_words]) for sentence in X_unlabelled]

# Split the data (for inside testing purposes)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Track training time
start_time = time.time()

# Train the model
pipeline.fit(X,y)

# Calculate and print elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Training completed in {elapsed_time:.2f} seconds.')

# Evaluate the model (for inside testing purposes)
# score = pipeline.score(X_test, y_test)
# print(f'Accuracy: {score:.4f}')

# Making predictions on unlabelled data
predictions = pipeline.predict(X_unlabelled)
print(predictions)

# new dataset with id from data_unlabelled and predicted difficulty
data_unlabelled['difficulty'] = predictions
data_unlabelled_submit = data_unlabelled[['id', 'difficulty']]
data_unlabelled_submit.to_csv(os.path.abspath("../outputs/datasets/predicted_embedding_nostopwords.csv"), index=False)