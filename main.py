from agents.monitoring_agent import MonitoringAgent
from agents.detection_agent import DetectionAgent
from agents.response_agent import ResponseAgent
import pandas as pd

DATA_PATH = "dataset/nsl_kdd_sample.csv"

monitor = MonitoringAgent(DATA_PATH)
detector = DetectionAgent()
responder = ResponseAgent()

model_ready = False  # ← Track if the model has been trained at least once

for row in monitor.stream_data():
    label_text = row["label"]
    label = 0 if label_text == "normal" else 1

    input_row = row.drop("label")
    X = pd.DataFrame([input_row])
    X_processed = detector.preprocess(X.copy())

    # 👇 Only predict after the first partial_fit
    if model_ready:
        prediction = detector.predict(X_processed)
        conf = detector.confidence(X_processed)
        print(f"[Prediction] → {'malicious' if prediction else 'normal'} (Confidence: {conf:.2f})")
        responder.take_action(prediction == 1, row)
    else:
        print("[*] Skipping prediction (model not yet trained)")

    # 👇 Train the model incrementally
    detector.train_incremental(X_processed, [label])
    model_ready = True  # ✅ Now it's safe to predict in the next loop
