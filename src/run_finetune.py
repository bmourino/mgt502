import os
import nltk
import pandas as pd
from embed import BertFineTuner
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('../available_datasets/training_data.csv')

X = df['sentence'].values
y = df['difficulty'].values

# remove stopwords
stop_words = set(stopwords.words('french'))
X = [' '.join([word for word in sentence.split() if word.lower() not in stop_words]) for sentence in X]

bert_fine_tuner = BertFineTuner(X, y)
bert_fine_tuner.train()

data_unlabelled = pd.read_csv(os.path.abspath("../available_datasets/unlabelled_test_data.csv"))
X_unlabelled = data_unlabelled['sentence'].values
X_unlabelled = [' '.join([word for word in sentence.split() if word.lower() not in stop_words]) for sentence in X_unlabelled]

predictions = bert_fine_tuner.predict(X_unlabelled)
print(predictions)

# new dataset with id from data_unlabelled and predicted difficulty
data_unlabelled['difficulty'] = predictions
data_unlabelled_submit = data_unlabelled[['id', 'difficulty']]
data_unlabelled_submit.to_csv(os.path.abspath("../outputs/datasets/predicted_bertfinetuner.csv"), index=False)