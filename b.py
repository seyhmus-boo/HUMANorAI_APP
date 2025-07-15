import pandas as pd
import openai
import time

#
gemini.api_key = "AIzaSyB0JxdKZzX0JIfZ5PNFTX3D576aXu_BSVg"

def generate_synthetic_abstract(field, max_tokens=150):
    prompt = (f"Write an academic abstract in the field of {field}. "
              "Include background, method, and conclusion sections. "
              "The style should mimic real research abstracts.")
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            n=1,
            stop=None,
        )
        abstract = response.choices[0].text.strip()
        return abstract
    except Exception as e:
        print(f"Error generating synthetic abstract: {e}")
        return ""

categories = ['Sociology', 'Medicine', 'Natural Sciences', 'Technology']

num_per_category = 5000  # Her kategori için sentetik örnek sayısı

synthetic_abstracts = []
labels = []

for field in categories:
    print(f"Generating synthetic abstracts for {field}...")
    for _ in range(num_per_category):
        abstract = generate_synthetic_abstract(field)
        synthetic_abstracts.append(abstract)
        labels.append(field)
        time.sleep(0.5)  # API limitlerini zorlamamak için

df = pd.DataFrame({
    'category': labels,
    'abstract': synthetic_abstracts
})

df.to_csv('C:/Users/90541/OneDrive - Manisa Celal Bayar Üniversitesi/Microsoft Copilot Chat Dosyaları/Masaüstü/AI', index=False)

print("Sentetik veriler oluşturuldu ve kaydedildi.")
