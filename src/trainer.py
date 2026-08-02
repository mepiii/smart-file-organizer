import os, re, pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def train_model(data_path, model_path):
    df = pd.read_csv(data_path)
    X = df['filename'].apply(lambda t: re.sub(r'[^a-zA-Z0-9]', ' ', t).lower())
    pipeline = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3), token_pattern=r'\w+')), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
    pipeline.fit(X, df['target_folder'])
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
