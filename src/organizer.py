import os, re, json, joblib
class SmartFileOrganizer:
    def __init__(self, m, c): self.pipeline = joblib.load(m)
    def predict_folder(self, f):
        c = re.sub(r'[^a-zA-Z0-9]', ' ', f).lower()
        p = self.pipeline.predict_proba([c])[0]
        return self.pipeline.classes_[p.argmax()], float(p.max())
