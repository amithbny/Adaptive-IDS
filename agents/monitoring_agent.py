import pandas as pd

class MonitoringAgent:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df = self.df.sample(frac=1).reset_index(drop=True)  # Shuffle rows

    def stream_data(self):
        for _, row in self.df.iterrows():
            yield row
