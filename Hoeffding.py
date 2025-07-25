import os
import sys
import signal
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import numpy as np
from river import tree, metrics, preprocessing

# === CONFIGURATION ===
LABELLED_FOLDER = r".\23_IND_Dataset\train"
SELECTED_FEATURES = ['avg', 'header_length', 'std', 'iat', 'ack_flag_number']


CHECKPOINT_PATH = r".\output\downsample3\downsample3.json"
MODEL_PATH = r".\output\Individual Test\23\online\Hoeffding\Hoeff_23_ind_train.joblib"
ACCURACY_DATA_PATH = r".\output\downsample3\downsample3.npz"
REPORT_PATH = r".\output\downsample3\downsample3.txt"
PLOT_PATH = r".\output\downsample3\downsample3.png"


trainer_global = None  # Global reference for signal handler

# === ONLINE TRAINER CLASS ===
class OnlineTrainer:
    def __init__(self):
        self.model = tree.HoeffdingTreeClassifier()
        self.scaler = preprocessing.MinMaxScaler()
        self.accuracy = metrics.Accuracy()
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
        self.f1 = metrics.F1()
        self.conf_matrix = metrics.ConfusionMatrix()
        self.accuracies = []
        self.data_points = []

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            self.model, self.scaler = joblib.load(MODEL_PATH)
            print("✅ Loaded model from checkpoint.")
        self.load_accuracy_data()

    def save_model(self):
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            tmp_path = MODEL_PATH + ".tmp"
            joblib.dump((self.model, self.scaler), tmp_path)
            os.replace(tmp_path, MODEL_PATH)  # Atomic replace
            self.save_accuracy_data()
            print("✅ Model and accuracy saved.")
        except Exception as e:
            print(f"❌ Failed to save model: {e}")

    def train_on_csv(self, csv_path, start_index=0):
        print(f"\n🚀 Training from: {os.path.basename(csv_path)} | Starting from row: {start_index}")
        df = pd.read_csv(csv_path)

        # Normalize column names to lowercase with underscores
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        if 'label' not in df.columns:
            print("⚠️ Skipping, no 'label' column found after normalization.")
            return


        df['label'] = df['label'].apply(lambda x: 0 if str(x).lower() == 'benign' else 1)
        df = df[[*SELECTED_FEATURES, 'label']].dropna().reset_index(drop=True)

        for i in range(start_index, len(df)):
            try:
                row = df.iloc[i]
                x_raw = row.drop('label').to_dict()
                y = row['label']
                self.scaler.learn_one(x_raw)
                x = self.scaler.transform_one(x_raw)
                y_pred = self.model.predict_one(x)

                if y_pred is not None:
                    #if i % 10000 == 0:  # Only evaluate every 10000 samples
                    self.accuracy.update(y, y_pred)
                    self.precision.update(y, y_pred)
                    self.recall.update(y, y_pred)
                    self.f1.update(y, y_pred)
                    self.conf_matrix.update(y, y_pred)
                    self.accuracies.append(self.accuracy.get())
                    self.data_points.append(len(self.data_points))

                self.model.learn_one(x, y)

                # Save checkpoint + model every 100000 samples
                if i % 100000 == 0:
                    save_checkpoint(os.path.basename(csv_path), i)
                    self.save_model()

            except Exception as e:
                print(f"⚠️ Error at row {i}: {e}")
                continue

    def get_stats(self):
        if not self.accuracies:
            return None, None
        return self.accuracies[-1], sum(self.accuracies) / len(self.accuracies)

    def save_accuracy_data(self):
        try:
            os.makedirs(os.path.dirname(ACCURACY_DATA_PATH), exist_ok=True)
            # Save as compressed numpy array
            np.savez_compressed(
                ACCURACY_DATA_PATH,
                data_points=np.array(self.data_points),
                accuracies=np.array(self.accuracies)
            )
        except Exception as e:
            print(f"⚠️ Could not save accuracy data: {e}")

    def load_accuracy_data(self):
        if os.path.exists(ACCURACY_DATA_PATH):
            try:
                data = np.load(ACCURACY_DATA_PATH)
                self.data_points = data['data_points'].tolist()
                self.accuracies = data['accuracies'].tolist()
            except Exception as e:
                print(f"⚠️ Could not load accuracy data: {e}")

    def save_report(self, path=REPORT_PATH):
        final_acc, avg_acc = self.get_stats()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            if final_acc is not None:
                f.write(f"Final Accuracy: {final_acc:.4f}\n")
                f.write(f"Average Accuracy: {avg_acc:.4f}\n\n")

            f.write("=== Classification Report ===\n")
            f.write(f"Accuracy: {self.accuracy.get():.4f}\n")
            f.write(f"Precision: {self.precision.get():.4f}\n")
            f.write(f"Recall (TPR): {self.recall.get():.4f}\n")
            f.write(f"F1 Score: {self.f1.get():.4f}\n\n")

            f.write("=== Confusion Matrix ===\n")
            f.write(str(self.conf_matrix) + "\n\n")

            try:
                cm = self.conf_matrix
                TP = cm[1][1]
                TN = cm[0][0]
                FP = cm[0][1]
                FN = cm[1][0]

                TPR = TP / (TP + FN) if (TP + FN) else 0
                TNR = TN / (TN + FP) if (TN + FP) else 0
                FPR = FP / (FP + TN) if (FP + TN) else 0
                FNR = FN / (FN + TP) if (FN + TP) else 0

                f.write("=== Extended Metrics ===\n")
                f.write(f"True Positive Rate (TPR): {TPR:.4f}\n")
                f.write(f"True Negative Rate (TNR): {TNR:.4f}\n")
                f.write(f"False Positive Rate (FPR): {FPR:.4f}\n")
                f.write(f"False Negative Rate (FNR): {FNR:.4f}\n")
            except Exception as e:
                f.write(f"Error computing extended metrics: {e}")

        print(f"📄 Report saved at {path}")

    def replot_graph(self,
                     color='green',
                     y_min=0.9,
                     y_max=1.0,
                     title='Accuracy Over Time',
                     figsize=(10, 5),
                     save_path=PLOT_PATH):
        if not self.data_points or not self.accuracies:
            print("⚠️ No data to plot.")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=figsize)
        plt.plot(self.data_points, self.accuracies, color=color)
        plt.xlabel("Samples Seen")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.ylim(y_min, y_max)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"✅ Plot saved at {save_path}")

# === CHECKPOINT UTILS ===
def save_checkpoint(current_file, current_index):
    try:
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump({"file": current_file, "index": current_index}, f)
    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
    return None

# === SIGNAL HANDLER FOR CTRL+C AND KILL ===
def signal_handler(signum, frame):
    print(f"\n⏸️ Signal {signum} received, saving model and checkpoint safely...")
    if trainer_global:
        trainer_global.save_model()
        print("✅ Model saved on interrupt.")
    else:
        print("⚠️ Trainer not initialized yet, skipping save.")
    sys.exit(0)  # Clean exit

# === MAIN EXECUTION ===
def main():
    global trainer_global
    trainer = OnlineTrainer()
    trainer_global = trainer  # Global reference for signal handling

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill command

    trainer.load_model()

    checkpoint = load_checkpoint()
    files = sorted(f for f in os.listdir(LABELLED_FOLDER) if f.endswith(".csv"))

    for file in files:
        start_index = 0
        if checkpoint and file == checkpoint["file"]:
            start_index = checkpoint["index"]
            checkpoint = None
        elif checkpoint:
            # Skip files before checkpoint file
            continue

        path = os.path.join(LABELLED_FOLDER, file)
        trainer.train_on_csv(path, start_index)
        trainer.save_model()

    trainer.save_report()
    trainer.replot_graph()

if __name__ == "__main__":
    main()