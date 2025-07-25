import pandas as pd
from river import anomaly, compose, preprocessing
import numpy as np
import joblib
from datetime import datetime
import os

# --- Parameters ---
BENIGN_CSV = r"Clustering\full_no_follow24.csv"
MODEL_PATH = r"Clustering\Models\full_no_follow24_hst.joblib"
FEATURES = ['avg', 'header_length', 'std', 'iat', 'ack_flag_number']

# --- Load Data with NaN Handling ---
def load_and_clean_data(path):
    df = pd.read_csv(path)
    df['label'] = df['label'].apply(lambda x: 0 if 'benign' in str(x).lower() else 1)
    for feat in FEATURES:
        if df[feat].isna().any():
            median_val = df[feat].median()
            df[feat] = df[feat].fillna(median_val)
            print(f"Filled NaN in {feat} with median: {median_val:.2f}")
    return df

df_benign = load_and_clean_data(BENIGN_CSV)
df_benign = df_benign[df_benign['label'] == 0]

# --- Model Pipeline ---
model = compose.Pipeline(
    preprocessing.MinMaxScaler(),
    anomaly.HalfSpaceTrees(
        n_trees=25,
        height=15,
        window_size=250,
        seed=42,
        limits={'avg': (0, 1000)}
    )
)

# --- Training ---
print("\nTraining on benign data...")
benign_scores = []
for _, row in df_benign.iterrows():
    x = {feat: row[feat] for feat in FEATURES}
    model.learn_one(x)
    score = model.score_one(x)
    benign_scores.append(score)

threshold = np.percentile(benign_scores, 99)
print(f"Anomaly threshold (99th percentile): {threshold:.2f}")

# --- Save Model and Threshold ---
model_data = {
    'model': model,
    'threshold': threshold,
    'features': FEATURES,
    'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
joblib.dump(model_data, MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
