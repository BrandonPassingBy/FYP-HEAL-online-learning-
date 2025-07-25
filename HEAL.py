import os
import sys
import signal
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import numpy as np
from river import tree, anomaly, metrics, preprocessing, active
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime
import pickle

# === CONFIGURATION ===
LABELLED_FOLDER = r"d:\Downsampling\CICIOMT2024\just_shuffling\train"
SELECTED_FEATURES = ['avg', 'header_length', 'std', 'iat', 'ack_flag_number']

CHECKPOINT_PATH = os.path.join("output", "SelfLabel2", "checkpoint.pkl")  # ✅ correct full file path
MODEL_PATH = r".\output\Individual Test\23\online\Hoeffding\Hoeff_23_ind_train.joblib" # original model path
PROGRESS_PATH = r"./output/SelfLabel2/Models/2324_base_model_progress.joblib"  # your training-in-progress version

HST_MODEL_PATH = r"hst_model.joblib"

ACCURACY_DATA_PATH = r".\output\SelfLabel2\Hoeff_SL_train.npz"
REPORT_PATH = r".\output\SelfLabel2\Downsample_quick_test.txt"
PLOT_PATH = r".\output\SelfLabel2\Downsample_quick_test.png"
TEST_REPORT_PATH = r".\output\SelfLabel2\Downsample_quick_test.txt"
TEST_LOADED_REPORT_PATH = r".\output\SelfLabel2\Downsample_quick_test_loaded.txt"

trainer_global = None  # Global reference for signal handler
is_test_case = False  # Flag to track test case execution


# === ONLINE TRAINER CLASS ===
class OnlineTrainer:
    def __init__(self, hst_threshold=0.5, discount_factor=2):
        self.model = tree.HoeffdingTreeClassifier()
        self.sampler = active.EntropySampler(classifier=self.model, discount_factor=discount_factor, seed=42)
        self.hst = anomaly.HalfSpaceTrees(n_trees=10, height=10, window_size=100, seed=42)
        self.hst_threshold = hst_threshold
        self.scaler = preprocessing.MinMaxScaler()
        self.accuracy = metrics.Accuracy()
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
        self.f1 = metrics.F1()
        self.conf_matrix = metrics.ConfusionMatrix()
        self.accuracies = []
        self.data_points = []
        self.entropies = []
        self.hst_queries = 0

    def load_data(self, csv_path, n_samples=None):
        """Load and preprocess CSV data."""
        try:
            df = pd.read_csv(csv_path)
            if n_samples:
                df = df.head(n_samples)
            print(f"🧾 CSV columns: {df.columns.tolist()}")

            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
            if 'label' not in df.columns:
                print("⚠️ Skipping, no 'label' column found.")
                return None

            df['label'] = df['label'].apply(lambda x: 0 if str(x).lower() == 'benign' else 1)
            for feat in SELECTED_FEATURES:
                if feat not in df.columns:
                    print(f"⚠️ Missing feature '{feat}' in CSV columns: {df.columns.tolist()}")
                    return None
                df[feat] = pd.to_numeric(df[feat], errors='coerce')
                if df[feat].isna().any():
                    median_val = df[feat].median()
                    df[feat] = df[feat].fillna(median_val)
                    print(f"ℹ️ Filled NaN in {feat} with median: {median_val:.2f}")

            return df[[*SELECTED_FEATURES, 'label']].dropna().reset_index(drop=True)
        except Exception as e:
            print(f"⚠️ Failed to load CSV: {e}")
            return None

    def validate_features(self, x_raw, row_idx):
        """Validate feature dictionary."""
        if not isinstance(x_raw, dict):
            print(f"⚠️ Row {row_idx}: x_raw is not a dictionary, type: {type(x_raw)}, value: {x_raw}")
            return False
        for feat in SELECTED_FEATURES:
            if feat not in x_raw:
                print(f"⚠️ Row {row_idx}: Missing feature '{feat}' in x_raw: {x_raw}")
                return False
            value = x_raw[feat]
            if not isinstance(value, (int, float)) or pd.isna(value):
                print(f"⚠️ Row {row_idx}: Non-numeric or NaN value for '{feat}': {value}")
                return False
        return True

    def update_metrics(self, y_true, y_pred):
        """Update evaluation metrics."""
        if y_pred is not None:
            self.accuracy.update(y_true, y_pred)
            self.precision.update(y_true, y_pred)
            self.recall.update(y_true, y_pred)
            self.f1.update(y_true, y_pred)
            self.conf_matrix.update(y_true, y_pred)
            self.accuracies.append(self.accuracy.get())
            self.data_points.append(len(self.data_points))

    def load_model(self, load_classifier=False):
        """Load HST model, and optionally classifier model."""
        if os.path.exists(HST_MODEL_PATH):
            hst_data = joblib.load(HST_MODEL_PATH)
            self.hst = hst_data['model']
            self.hst_threshold = hst_data.get('threshold', 0.5)
            print("✅ Loaded HalfSpaceTrees model from checkpoint.")
        else:
            print(f"⚠️ HST model not found at {HST_MODEL_PATH}. Using initialized model.")

        if load_classifier and os.path.exists(MODEL_PATH):
            try:
                self.model, self.scaler = joblib.load(MODEL_PATH)
                self.sampler = active.EntropySampler(classifier=self.model, discount_factor=self.sampler.discount_factor, seed=42)
                print(f"✅ Loaded HoeffdingTreeClassifier from {MODEL_PATH}.")
            except Exception as e:
                print(f"⚠️ Failed to load classifier model: {e}")
                self.model = tree.HoeffdingTreeClassifier()
                self.sampler = active.EntropySampler(classifier=self.model, discount_factor=self.sampler.discount_factor, seed=42)
                self.scaler = preprocessing.MinMaxScaler()
        elif not load_classifier:
            print(f"ℹ️ HoeffdingTreeClassifier model at {MODEL_PATH} ignored for test case.")



    def save_model(self, suffix=None):
        """Save the current model to a consistent progress path (overwrite each time)."""
        if is_test_case:
            print("⚠️ Attempt to save model during test case ignored to keep HST model fixed.")
            return
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

            # === Save to fixed progress path ===
            progress_path = MODEL_PATH.replace(".joblib", "_progress.joblib")
            # hst_progress_path = HST_MODEL_PATH.replace(".joblib", "_progress.joblib")  # optional

            joblib.dump((self.model, self.scaler), progress_path)
            # joblib.dump({'model': self.hst, 'threshold': self.hst_threshold}, hst_progress_path)  # optional
            self.save_accuracy_data()

            print(f"✅ Overwrote progress model at:\n - {progress_path}")

        except Exception as e:
            print(f"❌ Failed to save progress model: {e}")




    def train_on_csv(self, csv_path, start_index=0, discount_factor=2, use_self_labeling=False):
        """Train on CSV data using HST labels only. Entropy is ignored."""
        print(f"\n🚀 Training from: {os.path.basename(csv_path)} | Starting from row: {start_index}")
        df = self.load_data(csv_path)
        if df is None:
            return

        # EntropySampler still initialized if needed later, but not used
        sampler = active.EntropySampler(classifier=self.model, discount_factor=discount_factor, seed=42) if use_self_labeling else None

        for i in range(start_index, len(df)):
            try:
                row = df.iloc[i]
                x_raw = row[SELECTED_FEATURES].to_dict()
                y_true = row['label']  # still used for logging only

                if not self.validate_features(x_raw, i):
                    continue

                self.scaler.learn_one(x_raw)
                x = self.scaler.transform_one(x_raw)
                if x is None:
                    print(f"⚠️ Row {i}: Scaler returned None for x_raw: {x_raw}")
                    continue

                # 🔹 Label only using HST
                score = self.hst.score_one(x)
                y = 1 if score > self.hst_threshold else 0
                self.hst_queries += 1

                # 🔹 Log prediction performance
                self.update_metrics(y_true, self.model.predict_one(x))

                # 🔹 Train classifier
                self.model.learn_one(x, y)

                if i % 100000 == 0:
                    save_checkpoint(os.path.basename(csv_path), i)
                    self.save_model()

            except Exception as e:
                print(f"⚠️ Error at row {i}: {e}")
                continue






    def reset_metrics(self):
        """Reset metrics and state."""
        self.model = tree.HoeffdingTreeClassifier()
        self.sampler = active.EntropySampler(classifier=self.model, discount_factor=self.sampler.discount_factor, seed=42)
        self.scaler = preprocessing.MinMaxScaler()
        self.accuracy = metrics.Accuracy()
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
        self.f1 = metrics.F1()
        self.conf_matrix = metrics.ConfusionMatrix()
        self.accuracies = []
        self.data_points = []
        self.entropies = []
        self.hst_queries = 0


    def _save_test_results(self, predictions, true_labels, report_path, n_samples):
        """Save test results to file."""
        accuracy = accuracy_score(true_labels, predictions)
        avg_entropy = np.mean(self.entropies) if self.entropies else 0
        query_rate = self.hst_queries / n_samples if n_samples > 0 else 0

        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(f"Test Case - Accuracy: {accuracy:.4f}\n")
            f.write(f"Test Case - Average Entropy: {avg_entropy:.4f}\n")
            f.write(f"Test Case - HST Queries: {self.hst_queries} ({query_rate*100:.2f}%)\n\n")
            f.write("=== Classification Report ===\n")
            f.write(classification_report(true_labels, predictions, target_names=['Benign', 'Attack']))

        print(f"📄 Test report saved at {report_path}")
        print(f"Test Case - Accuracy: {accuracy:.4f}")
        print(f"Test Case - Average Entropy: {avg_entropy:.4f}")
        print(f"Test Case - HST Queries: {self.hst_queries} ({query_rate*100:.2f}%)")

    def save_accuracy_data(self):
        """Save accuracy data."""
        if is_test_case:
            print("⚠️ Attempt to save accuracy data during test case ignored.")
            return
        try:
            os.makedirs(os.path.dirname(ACCURACY_DATA_PATH), exist_ok=True)
            np.savez_compressed(
                ACCURACY_DATA_PATH,
                data_points=np.array(self.data_points),
                accuracies=np.array(self.accuracies),
                entropies=np.array(self.entropies),
                hst_queries=np.array([self.hst_queries])
            )
        except Exception as e:
            print(f"⚠️ Could not save accuracy data: {e}")

    def load_accuracy_data(self):
        """Load accuracy data."""
        if os.path.exists(ACCURACY_DATA_PATH):
            try:
                data = np.load(ACCURACY_DATA_PATH)
                self.data_points = data['data_points'].tolist()
                self.accuracies = data['accuracies'].tolist()
                self.entropies = data.get('entropies', []).tolist()
                self.hst_queries = int(data.get('hst_queries', [0])[0])
            except Exception as e:
                print(f"⚠️ Could not load accuracy data: {e}")

    def save_report(self, path=REPORT_PATH):
        """Save full training report."""
        final_acc, avg_acc = self.get_stats()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            if final_acc is not None:
                f.write(f"Final Accuracy: {final_acc:.4f}\n")
                f.write(f"Average Accuracy: {avg_acc:.4f}\n")
                f.write(f"Average Entropy: {np.mean(self.entropies):.4f}\n")
                f.write(f"HST Queries: {self.hst_queries}\n\n")

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
                f.write(f"True Positive Rate (TPR): {abs(TPR):.4f}\n")
                f.write(f"True Negative Rate: {abs(TNR):.4f}\n")
                f.write(f"False Positive Rate: {abs(FPR):.2f}\n")
                f.write(f"False Negative Rate: {abs(FNR):.1f}\n")
            except Exception as e:
                f.write(f"Error computing extended metrics: {e}")

        print(f"📄 Report saved at {path}")

    def get_stats(self):
        """Get final and average accuracy."""
        if not self.accuracies:
            return None, None
        return self.accuracies[-1], sum(self.accuracies) / len(self.accuracies)

    def replot_graph(self,
                     color='red',
                     y_min=0.1,
                     y_max=1.0,
                     title='Accuracy Over Time',
                     figsize=(20, 10),
                     save_path=PLOT_PATH):
        """Plot accuracy and entropy."""
        if not self.data_points or not self.accuracies:
            print("⚠️ No test data to plot.")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=figsize)
        plt.plot(self.data_points, self.accuracies, color=color, label='Accuracy')
        if self.entropies:
            plt.plot(self.data_points[:len(self.entropies)], self.entropies, color='blue', label='Entropy')
        plt.xlabel("Test Samples")
        plt.ylabel("Accuracy / Entropy")
        plt.title(title)
        plt.ylim(y_min, y_max)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close()

# === CHECKPOINT UTILS ===
def save_checkpoint(current_file, index):
    """Save training checkpoint."""
    try:
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        with open(CHECKPOINT_PATH, 'w') as file:
            json.dump({"file": current_file, "index": index}, file)
    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")

def load_checkpoint():
    try:
        if not os.path.exists(CHECKPOINT_PATH):
            print("📭 No checkpoint found. Starting from scratch.")
            return None

        with open(CHECKPOINT_PATH, "rb") as f:
            checkpoint = pickle.load(f)
        print(f"✅ Loaded checkpoint: {checkpoint}")
        return checkpoint

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return None

# === SIGNAL HANDLER ===
def signal_handler(signum, frame):
    global is_test_case
    print(f"\n⏸️ Received signal {signum}, handling safely...")
    if trainer_global and not is_test_case:
        trainer_global.save_model()
        print("✅ Model saved on interrupt (full training mode).")
    else:
        print("ℹ️ No model saving during test case or trainer not initialized.")
    sys.exit(0)

# === MAIN EXECUTION ===
def main():
    global trainer_global
    trainer = OnlineTrainer(discount_factor=2, hst_threshold=0.5)
    trainer_global = trainer

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


    test_csv = os.path.join(LABELLED_FOLDER, sorted(os.listdir(LABELLED_FOLDER))[0])

    '''
    print("🧪 Running original test case...")
    trainer.test_case(test_csv, train_samples=700, eval_samples=700, hst_threshold=0.87)
    print("✅ Original test case completed.")
    '''

    '''
    print("\n🧪 Running loaded model test case...")
    trainer.test_case_with_loaded_model(test_csv, train_samples=20, eval_samples=400, discount_factor=5, hst_threshold=0.87)
    print("✅ Loaded model test case completed.")
    '''
    # Full training (uncomment to run)
    
    print("🚀 Running full training with self-labeling...")
    trainer.load_model()
    checkpoint = load_checkpoint()
    files = sorted(f for f in os.listdir(LABELLED_FOLDER) if f.endswith(".csv"))

    for file in files:
        start_index = 0
        if checkpoint and file == checkpoint["file"]:
            start_index = checkpoint["index"]
            checkpoint = None
        elif checkpoint:
            continue

        path = os.path.join(LABELLED_FOLDER, file)
        trainer.train_on_csv(path, start_index, discount_factor=0, use_self_labeling=False)
        trainer.save_model()

    trainer.save_report()
    trainer.replot_graph()
    print("✅ Full training completed.")
    

if __name__ == "__main__":
    main()