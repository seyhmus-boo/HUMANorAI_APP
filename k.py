# GEREKLİ KÜTÜPHANELER
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from transformers import (
    BertTokenizer, TFBertForSequenceClassification,
    RobertaTokenizer, TFRobertaForSequenceClassification,
    AutoTokenizer, AutoModelForSeq2SeqLM,
    logging as hf_logging
)
from autogluon.text import TextPredictor
import gradio as gr
import warnings
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# NLTK
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# VERİ YÜKLEME VE ÖN İŞLEME
df = pd.read_csv("C:/Users/90541/OneDrive - Manisa Celal Bayar Üniversitesi/Microsoft Copilot Chat Dosyaları/Masaüstü/yeni_dataset.csv")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])

df['clean_text'] = df['text'].apply(preprocess_text)

# LABEL ENCODING
label_encoder = LabelEncoder()
df['label_enc'] = label_encoder.fit_transform(df['label'])

# EĞİTİM VE TEST AYIRMA
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label_enc'], test_size=0.2, random_state=42)

# VECTORIZER
vectorizer_bow = CountVectorizer()
vectorizer_tfidf = TfidfVectorizer()
X_train_bow = vectorizer_bow.fit_transform(X_train)
X_test_bow = vectorizer_bow.transform(X_test)
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

# ML MODELLERİ
ml_models_bow = {
    'Logistic Regression (BoW)': LogisticRegression(max_iter=1000),
    'Naive Bayes (BoW)': MultinomialNB(),
    'Random Forest (BoW)': RandomForestClassifier(),
    'SVM (BoW)': SVC(probability=True)
}

ml_models_tfidf = {
    'Logistic Regression (TF-IDF)': LogisticRegression(max_iter=1000),
    'Naive Bayes (TF-IDF)': MultinomialNB(),
    'Random Forest (TF-IDF)': RandomForestClassifier(),
    'SVM (TF-IDF)': SVC(probability=True)
}

print("--- TRAINING ML MODELS (BoW) ---")
for name, model in ml_models_bow.items():
    model.fit(X_train_bow, y_train)
    preds = model.predict(X_test_bow)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))

print("--- TRAINING ML MODELS (TF-IDF) ---")
for name, model in ml_models_tfidf.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))

# LSTM
print("--- TRAINING LSTM MODEL ---")
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
max_len = 200
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

lstm_model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_len),
    LSTM(128),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])
lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_train_pad, y_train, epochs=3, batch_size=32, validation_data=(X_test_pad, y_test))
lstm_preds = lstm_model.predict(X_test_pad).argmax(axis=1)
acc_lstm = accuracy_score(y_test, lstm_preds)
print(f"LSTM Accuracy: {acc_lstm:.4f}")
print(classification_report(y_test, lstm_preds, target_names=label_encoder.classes_))

# BERT
print("--- TRAINING BERT MODEL ---")
tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))
def bert_encode(texts, tokenizer, max_len=128):
    return tokenizer(list(texts), padding='max_length', truncation=True, max_length=max_len, return_tensors="tf")
X_train_enc = bert_encode(X_train, tokenizer_bert)
X_test_enc = bert_encode(X_test, tokenizer_bert)
bert_model.compile(optimizer=Adam(learning_rate=3e-5),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
bert_model.fit(x={"input_ids": X_train_enc['input_ids'], "attention_mask": X_train_enc['attention_mask']},
               y=y_train,
               validation_data=({"input_ids": X_test_enc['input_ids'], "attention_mask": X_test_enc['attention_mask']}, y_test),
               epochs=3, batch_size=16)
bert_preds = bert_model.predict({"input_ids": X_test_enc["input_ids"], "attention_mask": X_test_enc["attention_mask"]}).logits.argmax(axis=1)
acc_bert = accuracy_score(y_test, bert_preds)
print(f"BERT Accuracy: {acc_bert:.4f}")
print(classification_report(y_test, bert_preds, target_names=label_encoder.classes_))

# RoBERTa
print("--- TRAINING RoBERTa MODEL ---")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = TFRobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label_encoder.classes_))
def roberta_encode(texts, tokenizer, max_len=128):
    return tokenizer(list(texts), padding='max_length', truncation=True, max_length=max_len, return_tensors="tf")
X_train_roberta = roberta_encode(X_train, roberta_tokenizer)
X_test_roberta = roberta_encode(X_test, roberta_tokenizer)
roberta_model.compile(optimizer=Adam(learning_rate=3e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
roberta_model.fit(x={"input_ids": X_train_roberta['input_ids'], "attention_mask": X_train_roberta['attention_mask']},
                  y=y_train,
                  validation_data=({"input_ids": X_test_roberta['input_ids'], "attention_mask": X_test_roberta['attention_mask']}, y_test),
                  epochs=3, batch_size=16)
roberta_preds = roberta_model.predict({"input_ids": X_test_roberta['input_ids'], "attention_mask": X_test_roberta['attention_mask']}).logits.argmax(axis=1)
acc_roberta = accuracy_score(y_test, roberta_preds)
print(f"RoBERTa Accuracy: {acc_roberta:.4f}")
print(classification_report(y_test, roberta_preds, target_names=label_encoder.classes_))

# FLAN-T5
print("--- FLAN-T5 TEXT GENERATION ---")
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
def flan_generate(prompt):
    inputs = flan_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = flan_model.generate(**inputs, max_new_tokens=50, do_sample=True, top_p=0.9, temperature=0.8)
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

# AutoGluon
print("--- TRAINING AutoGluon MODEL ---")
df_ag = df[['clean_text', 'label']].rename(columns={'clean_text': 'text', 'label': 'label'})
train_data_ag, test_data_ag = train_test_split(df_ag, test_size=0.2, random_state=42)
ag_model = TextPredictor(label='label')
ag_model.fit(train_data_ag)
ag_preds = ag_model.predict(test_data_ag)
acc_ag = accuracy_score(test_data_ag['label'], ag_preds)
print(f"AutoGluon Accuracy: {acc_ag:.4f}")
print(classification_report(test_data_ag['label'], ag_preds))

# Gradio Arayüzü
model_list = list(ml_models_bow.keys()) + list(ml_models_tfidf.keys()) + ["LSTM", "BERT", "RoBERTa", "FLAN-T5", "AutoGluon"]

def predict_with_model(text, model_name):
    cleaned = preprocess_text(text)
    if "BoW" in model_name:
        vec = vectorizer_bow.transform([cleaned])
        model = ml_models_bow[model_name]
        pred = model.predict(vec)[0]
        return label_encoder.inverse_transform([pred])[0]
    elif "TF-IDF" in model_name:
        vec = vectorizer_tfidf.transform([cleaned])
        model = ml_models_tfidf[model_name]
        pred = model.predict(vec)[0]
        return label_encoder.inverse_transform([pred])[0]
    elif model_name == "LSTM":
        seq = tokenizer.texts_to_sequences([cleaned])
        pad = pad_sequences(seq, maxlen=max_len)
        pred = lstm_model.predict(pad).argmax(axis=1)[0]
        return label_encoder.inverse_transform([pred])[0]
    elif model_name == "BERT":
        enc = bert_encode([text], tokenizer_bert)
        pred = bert_model.predict({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}).logits.argmax(axis=1)[0]
        return label_encoder.inverse_transform([pred])[0]
    elif model_name == "RoBERTa":
        enc = roberta_encode([text], roberta_tokenizer)
        pred = roberta_model.predict({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}).logits.argmax(axis=1)[0]
        return label_encoder.inverse_transform([pred])[0]
    elif model_name == "FLAN-T5":
        return flan_generate(text)
    elif model_name == "AutoGluon":
        pred = ag_model.predict({"text": [text]})[0]
        return pred
    else:
        return "Model not found."

gr.Interface(
    fn=predict_with_model,
    inputs=[
        gr.Textbox(label="Text Input"),
        gr.Dropdown(choices=model_list, label="Select Model")
    ],
    outputs=gr.Textbox(label="Prediction / Output"),
    title="Text Classification and Generation System",
    description="You can perform text classification with different models or generate text with FLAN-T5."
).launch()
