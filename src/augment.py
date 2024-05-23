import nltk
import numpy as np
from nltk.corpus import stopwords
import nlpaug.augmenter.word as naw
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

# Data augmentation (tokens, cognates)
def augment_features(sentence):
    tokens = word_tokenize(sentence)
    num_tokens = len(tokens)
    num_cognates = sum(1 for token in tokens if token.lower() in stopwords.words('french'))
    return np.array([num_tokens, num_cognates])

class Augmenter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        augmented_features = np.array([augment_features(sentence) for sentence in X])
        return augmented_features

class TextAugmenter(BaseEstimator, TransformerMixin):
    def __init__(self, augmenter=None):
        self.augmenter = augmenter if augmenter else naw.SynonymAug(aug_src='wordnet')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        augmented_sentences = [self.augmenter.augment(sentence) for sentence in X]
        return [item for sublist in augmented_sentences for item in sublist]