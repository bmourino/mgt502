import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
import nlpaug.augmenter.word as naw
import pandas as pd

# Text embeddings using BERT
class BertEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-multilingual-cased', max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.max_length = max_length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        embeddings = []
        for sentence in tqdm(X, desc="Processing BERT embeddings"):
            inputs = self.tokenizer(sentence, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        return np.array(embeddings)

# Fine-tuning BERT 
class BertFineTuner(): # old: bert-base-multilingual-cased
    def __init__(self, sentences, labels, model_name='distilbert-base-multilingual-cased', max_length=128, batch_size=16, epochs=3, learning_rate=5e-5):
        self.sentences = sentences
        self.labels = labels
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=len(set(labels))).to(self.device)

        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

        self.train_dataloader, self.val_dataloader = self.prepare_data()

    def prepare_data(self):
        class CustomDataset(Dataset):
            def __init__(self, sentences, labels, tokenizer, max_length):
                self.sentences = sentences
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.sentences)

            def __getitem__(self, idx):
                sentence = self.sentences[idx]
                label = self.labels[idx]
                encoding = self.tokenizer(sentence, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }

        X_train, X_val, y_train, y_val = train_test_split(self.sentences, self.encoded_labels, test_size=0.2, random_state=42)

        train_dataset = CustomDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = CustomDataset(X_val, y_val, self.tokenizer, self.max_length)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)

        return train_dataloader, val_dataloader

    def train(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            correct_predictions = 0

            for batch in self.train_dataloader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()
                correct_predictions += torch.sum(torch.argmax(logits, dim=1) == labels)

                loss.backward()
                optimizer.step()

            avg_train_loss = total_loss / len(self.train_dataloader)
            train_accuracy = correct_predictions.double() / len(self.train_dataloader.dataset)

            val_loss, val_accuracy = self.evaluate()

            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Train loss: {avg_train_loss:.4f}, Train accuracy: {train_accuracy:.4f}")
            print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}")

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()
                correct_predictions += torch.sum(torch.argmax(logits, dim=1) == labels)

        avg_val_loss = total_loss / len(self.val_dataloader)
        val_accuracy = correct_predictions.double() / len(self.val_dataloader.dataset)

        return avg_val_loss, val_accuracy

    def predict(self, sentences):
        self.model.eval()
        encoded_sentences = self.tokenizer(sentences, truncation=True, padding=True, return_tensors='pt', max_length=self.max_length).to(self.device)
        input_ids = encoded_sentences['input_ids']
        attention_mask = encoded_sentences['attention_mask']

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        return self.label_encoder.inverse_transform(predictions)
    
class CustomDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = str(self.sentences[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'sentence_text': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertFineTuner2:
    def __init__(self, X, y, model_name='bert-base-multilingual-cased', max_len=128, batch_size=16, epochs=4, learning_rate=2e-5):
        self.X = X
        self.y = y
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(set(y)))

        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(y)

    def augment_data(self, sentences):
        augmenter = naw.SynonymAug(aug_src='wordnet')
        augmented_sentences = [augmenter.augment(sentence) for sentence in sentences]
        return augmented_sentences

    def preprocess_data(self):
        augmented_sentences = self.augment_data(self.X)
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(self.X)
        tfidf_features = tfidf_vectorizer.transform(augmented_sentences).toarray()

        scaler = MaxAbsScaler()
        scaled_features = scaler.fit_transform(tfidf_features)
        
        return scaled_features

    def prepare_data_loader(self, X, y):
        dataset = CustomDataset(
            sentences=X,
            labels=y,
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4
        )

    def train(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y_encoded, test_size=0.2, random_state=42)
        
        train_data_loader = self.prepare_data_loader(X_train, y_train)
        val_data_loader = self.prepare_data_loader(X_val, y_val)

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            self.model.train()
            for batch in train_data_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(torch.long)
                attention_mask = batch['attention_mask'].to(torch.long)
                labels = batch['labels'].to(torch.long)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {loss.item()}")

    def evaluate(self, data_loader):
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(torch.long)
                attention_mask = batch['attention_mask'].to(torch.long)
                labels = batch['labels'].to(torch.long)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                _, preds = torch.max(outputs.logits, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.size(0)

        accuracy = correct_predictions.double() / total_predictions
        return accuracy.item()
    
    def predict(self, sentences):
        self.model.eval()
        input_ids = []
        attention_masks = []

        for sentence in sentences:
            encoding = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids.append(encoding['input_ids'].flatten())
            attention_masks.append(encoding['attention_mask'].flatten())

        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_masks
            )

        _, preds = torch.max(outputs.logits, dim=1)
        predicted_labels = self.label_encoder.inverse_transform(preds.cpu().numpy())
        return predicted_labels