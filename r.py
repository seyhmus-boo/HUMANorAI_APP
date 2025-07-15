import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import textstat
import numpy as np

# NLTK indir
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# 1. Veriyi yükleme
df = pd.read_csv("C:/Users/90541/OneDrive - Manisa Celal Bayar Üniversitesi/Microsoft Copilot Chat Dosyaları/Masaüstü/yeni_dataset.csv")

# 2. Ön işleme
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text


df['clean_text'] = df['text'].apply(preprocess_text)


# Keyword yoğunluğu fonksiyonu
def calculate_keyword_density(text):
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha() and w not in stop_words]
    if not words:
        return 0, ''
    freq = pd.Series(words).value_counts()
    density = (freq.iloc[0] / len(words)) * 100
    return round(density, 2), freq.index[0]


# Flesch okunabilirlik skoru için wrapper (hatalı metinlerde sıfır döner)
def safe_flesch(text):
    try:
        return textstat.flesch_reading_ease(text)
    except:
        return 0


# Metrikleri hesapla ve ekle
df['flesch_score'] = df['text'].apply(safe_flesch)
df['keyword_density'], df['top_keyword'] = zip(*df['text'].apply(calculate_keyword_density))

# Veriyi hazırla
X = df['clean_text']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vektörleştiriciler
vectorizer_bow = CountVectorizer()
X_train_bow = vectorizer_bow.fit_transform(X_train)
X_test_bow = vectorizer_bow.transform(X_test)

vectorizer_tfidf = TfidfVectorizer()
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

results = []

for vectorizer_name, X_train_vec, X_test_vec in [('BoW', X_train_bow, X_test_bow),
                                                 ('TF-IDF', X_train_tfidf, X_test_tfidf)]:
    print(f"\n{vectorizer_name} ile Model Performans Değerlendirmesi:")

    # Cosine similarity ortalamasını hesapla (test setinde)
    cosine_sim = cosine_similarity(X_test_vec)
    avg_cosine_sim = np.mean(cosine_sim)

    # Flesch, keyword_density ortalamaları test setinde
    # Test indekslerini bul
    test_indices = X_test.index
    avg_flesch = df.loc[test_indices, 'flesch_score'].mean()
    avg_keyword_density = df.loc[test_indices, 'keyword_density'].mean()

    for model_name, model in models.items():
        print(f"\n{model_name} Modeli:")

        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Doğruluk Oranı: {accuracy * 100:.2f}%")

        print("\nSınıflandırma Raporu:")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred, labels=y.unique())
        print("\nConfusion Matrix:")
        print(cm)

        auc = None
        if len(y.unique()) == 2:
            y_prob = model.predict_proba(X_test_vec)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            print(f"ROC AUC: {auc:.2f}")

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f"{model_name} {vectorizer_name} ROC Curve")
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.show()

        # Sonuçları yazdır
        print(f"\nOrtalama Flesch Reading Ease (test seti): {avg_flesch:.2f}")
        print(f"Ortalama Keyword Yoğunluğu % (test seti): {avg_keyword_density:.2f}")
        print(f"Ortalama Cosine Similarity (test seti): {avg_cosine_sim:.4f}")

        results.append({
            'Vectorizer': vectorizer_name,
            'Model': model_name,
            'Accuracy': accuracy * 100,
            'ROC AUC': auc,
            'Avg Flesch': avg_flesch,
            'Avg Keyword Density %': avg_keyword_density,
            'Avg Cosine Similarity': avg_cosine_sim
        })

results_df = pd.DataFrame(results)
print("\nModel Performans Sonuçları:")
print(results_df)
