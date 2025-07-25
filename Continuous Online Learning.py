import os
import sys
import signal
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import numpy as np
from river import tree, metrics, preprocessing
from datetime import datetime  # For optional multiple .npz files

# === CONFIGURATION ===
LABELLED_FOLDER = r"d:\Downsampling\CICIOMT2024\just_shuffling\train"
SELECTED_FEATURES = ['avg', 'header_length', 'std', 'iat', 'ack_flag_number']

# Default paths for adapted model
CHECKPOINT_PATH = r".\output\Adapt\Hoeff_23_adapted_24.json"
MODEL_PATH = r".\output\Adapt\Hoeff_23_adapted_24.joblib"
ACCURACY_DATA_PATH = r".\output\Adapt\Hoeff_23_adapted_24.npz"
REPORT_PATH = r".\output\Adapt\Hoeff_23_adapted_24.txt"
PLOT_PATH = r".\output\Adapt\Hoeff_23_adapted_24.png"

trainer_global = None  # Global reference for signal handler

# === ONLINE TRAINER CLASS ===
class OnlineTrainer:
    def __init__(self, input_model_path=None, output_model_path=MODEL_PATH, output_checkpoint_path=CHECKPOINT_PATH,
                 output_accuracy_path=ACCURACY_DATA_PATH, output_report_path=REPORT_PATH, output_plot_path=PLOT_PATH):
        self.model = tree.HoeffdingTreeClassifier()
        self.scaler = preprocessing.MinMaxScaler()
        self.accuracy = metrics.Accuracy()
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
        self.f1 = metrics.F1()
        self.conf_matrix = metrics.ConfusionMatrix()
        self.accuracies = []
        self.data_points = []
        self.input_model_path = input_model_path
        self.output_model_path = output_model_path
        self.output_checkpoint_path = output_checkpoint_path
        self.output_accuracy_path = output_accuracy_path
        self.output_report_path = output_report_path
        self.output_plot_path = output_plot_path

    def load_model(self):
        if self.input_model_path and os.path.exists(self.input_model_path):
            try:
                self.model, self.scaler = joblib.load(self.input_model_path)
                print(f"✅ Loaded model from: {self.input_model_path}")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
        self.load_accuracy_data()

    def save_model(self):
        try:
            os.makedirs(os.path.dirname(self.output_model_path), exist_ok=True)
            tmp_path = self.output_model_path + ".tmp"
            joblib.dump((self.model, self.scaler), tmp_path)
            os.replace(tmp_path, self.output_model_path)  # Atomic replace
            self.save_accuracy_data()
            print(f"✅ Model and accuracy saved to: {self.output_model_path}")
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
                self.scaler.learn_one(x_raw)  # Fixed: Learn scaler before transform
                x = self.scaler.transform_one(x_raw)
                y_pred = self.model.predict_one(x)

                if y_pred is not None:
                    if i % 10000 == 0:  # Evaluate every 10000 samples
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
                    save_checkpoint(os.path.basename(csv_path), i, checkpoint_path=self.output_checkpoint_path)
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
            os.makedirs(os.path.dirname(self.output_accuracy_path), exist_ok=True)
            # Save all data points (no truncation)
            np.savez_compressed(
                self.output_accuracy_path,
                data_points=np.array(self.data_points),
                accuracies=np.array(self.accuracies)
            )
            # Optional: Multiple .npz files per checkpoint (uncomment to enable)
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # npz_path = self.output_accuracy_path.replace(".npz", f"_{timestamp}.npz")
            # np.savez_compressed(npz_path, data_points=np.array(self.data_points), accuracies=np.array(self.accuracies))
            # print(f"✅ Saved accuracy data to: {npz_path}")
        except Exception as e:
            print(f"⚠️ Could not save accuracy data: {e}")

    def load_accuracy_data(self):
        if os.path.exists(self.output_accuracy_path):
            try:
                data = np.load(self.output_accuracy_path)
                self.data_points = data['data_points'].tolist()
                self.accuracies = data['accuracies'].tolist()
            except Exception as e:
                print(f"⚠️ Could not load accuracy data: {e}")

    def save_report(self, path=None):
        if path is None:
            path = self.output_report_path
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

    def replot_graph(self, color='green', y_min=0.9, y_max=1.0, title='Accuracy Over Time', figsize=(10, 5), save_path=None):
        if save_path is None:
            save_path = self.output_plot_path
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
def save_checkpoint(current_file, current_index, checkpoint_path=CHECKPOINT_PATH):
    try:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump({"file": current_file, "index": current_index}, f)
    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")

def load_checkpoint(checkpoint_path=CHECKPOINT_PATH):
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
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
    # Specify the CICIoT2023 model path
    cic_iot_model_path = r"C:\Users\brand\OneDrive\Documents\Sunway\FYP\Codes\output\Individual Test\23\online\Hoeffding\Hoeff_23_ind_train.joblib"
    
    # Initialize trainer with input and output paths
    trainer = OnlineTrainer(
        input_model_path=cic_iot_model_path,
        output_model_path=MODEL_PATH,
        output_checkpoint_path=CHECKPOINT_PATH,
        output_accuracy_path=ACCURACY_DATA_PATH,
        output_report_path=REPORT_PATH,
        output_plot_path=PLOT_PATH
    )
    trainer_global = trainer  # Global reference for signal handling

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill command

    trainer.load_model()

    checkpoint = load_checkpoint(checkpoint_path=trainer.output_checkpoint_path)
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