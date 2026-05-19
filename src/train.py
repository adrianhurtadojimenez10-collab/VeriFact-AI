import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Iniciando el proceso de entrenamiento de VeriFact-AI...")

try:
    df_true = pd.read_csv('data/True.csv')
    df_fake = pd.read_csv('data/Fake.csv')
except FileNotFoundError:
    print("Error: No se encontraron los archivos True.csv o Fake.csv en la carpeta 'data/'.")
    exit()

df_true['label'] = 1
df_fake['label'] = 0

df = pd.concat([df_true, df_fake], axis=0)
df = df[['text', 'label']].dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"📊 Dataset combinado con éxito. Total de noticias: {len(df)}")

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

print("Convirtiendo texto en vectores numéricos...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("Entrenando el modelo de Machine Learning...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"¡Entrenamiento completado! Precisión: {accuracy * 100:.2f}%")

print("Guardando el modelo entrenado...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/fake_news_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
print("Proceso finalizado. El 'cerebro' está listo en 'models/'.")
