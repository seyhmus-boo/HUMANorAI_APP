import pandas as pd
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # LaTeX and symbol removal
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    text = re.sub(r'\$\$.*?\$\$', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', ' ', text)
    text = re.sub(r'\{.*?\}', ' ', text)

    # Remove special characters except letters and space
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Remove digits and punctuation
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    tokens = text.split()

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Remove duplicate
    filtered_tokens = []
    prev_token = None
    for token in tokens:
        if token != prev_token:
            filtered_tokens.append(token)
        prev_token = token

    return ' '.join(filtered_tokens)


def load_and_label_data(folder_path, label):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df['source'] = label  # 'human' veya 'AI' etiketi
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


# Verilen klasör yolları
human_folder = r"C:\Users\90541\OneDrive - Manisa Celal Bayar Üniversitesi\Microsoft Copilot Chat Dosyaları\Masaüstü\human"
ai_folder = r"C:\Users\90541\OneDrive - Manisa Celal Bayar Üniversitesi\Microsoft Copilot Chat Dosyaları\Masaüstü\AI"

print("Human verileri yükleniyor...")
df_human = load_and_label_data(human_folder, 'human')
print("AI verileri yükleniyor...")
df_ai = load_and_label_data(ai_folder, 'AI')

print("Veriler birleştiriliyor...")
df = pd.concat([df_human, df_ai], ignore_index=True)

# Metin sütunu adı kontrolü
if 'text' in df.columns:
    text_col = 'text'
elif 'abstract' in df.columns:
    text_col = 'abstract'
else:
    raise ValueError("Dataset içinde 'text' veya 'abstract' sütunu bulunamadı.")

print("Metinler temizleniyor")
df['clean_text'] = df[text_col].astype(str).apply(preprocess_text)

print("Veri yeni_dataset.csv olarak kaydediliyor...")
df.to_csv('C:/Users/90541/OneDrive - Manisa Celal Bayar Üniversitesi/Microsoft Copilot Chat Dosyaları/Masaüstü/yeni_dataset.csv', index=False)

print("İşlem tamamlandı! yeni_dataset.csv dosyası oluştu.")
