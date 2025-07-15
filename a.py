import requests
import xml.etree.ElementTree as ET
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def build_query(keywords):
    query = " OR ".join([f'"{word}"' for word in keywords])
    return query

def get_arxiv_papers_with_keywords(keywords, max_results=500):
    base_url = 'http://export.arxiv.org/api/query?'
    query_str = build_query(keywords)
    url = f'{base_url}search_query=all:({query_str})&start=0&max_results={max_results}'
    response = requests.get(url)
    root = ET.fromstring(response.content)

    papers = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
        papers.append(title + " " + summary)
    return papers

def preprocess_text(text):
    text = text.lower()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Anahtar kelimeler
category_keywords = {
    'sociology': ['social', 'society', 'culture', 'behavior', 'community'],
    'medicine': ['health', 'disease', 'treatment', 'clinical', 'patient'],
    'natural_sciences': ['physics', 'chemistry', 'biology', 'environment', 'energy'],
    'technology': ['computer', 'algorithm', 'software', 'network', 'data']
}

all_papers = []
all_labels = []

for cat, keywords in category_keywords.items():
    print(f"Fetching papers for category: {cat}")
    papers = get_arxiv_papers_with_keywords(keywords, max_results=500)
    all_papers.extend(papers)
    all_labels.extend([cat]*len(papers))

# DataFrame oluşturma
df = pd.DataFrame({
    'category': all_labels,
    'text': all_papers
})


df['clean_text'] = df['text'].apply(preprocess_text)

print(df.head())
print(f"Total papers fetched: {len(df)}")


df.to_csv('C:/Users/90541/OneDrive - Manisa Celal Bayar Üniversitesi/Microsoft Copilot Chat Dosyaları/Masaüstü/human', index=False)
