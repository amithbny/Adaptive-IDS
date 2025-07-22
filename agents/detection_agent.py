import os
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

class DetectionAgent:
    def __init__(self, model_path='model/model.pkl'):
        self.model_path = model_path
        self.model = None
        self.label_encoders = {}
        self.classes = [0, 1]  # 0 = normal, 1 = malicious
        self.load_or_initialize_model()

    def load_or_initialize_model(self):
        if os.path.exists(self.model_path):
            self.model = load(self.model_path)
            print("[+] Loaded existing model.")
        else:
            self.model = SGDClassifier(loss='log_loss', warm_start=True)
            print("[*] Initialized new adaptive model.")

    def preprocess(self, df):
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        return df

    def train_incremental(self, X, y):
        self.model.partial_fit(X, y, classes=self.classes)
        dump(self.model, self.model_path)

    def predict(self, X):
        return self.model.predict(X)[0]

    def confidence(self, X):
        if hasattr(self.model, "predict_proba"):
            return max(self.model.predict_proba(X)[0])
        else:
            return 1.0  # fallback for classifiers that don't support predict_proba
