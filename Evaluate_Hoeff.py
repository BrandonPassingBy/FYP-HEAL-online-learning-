#!/usr/bin/env python3
"""
Evaluate.py   –  Evaluate a trained River model OR regenerate
                 the report/plot from a previously saved state.

Usage examples
--------------
# Normal evaluation (default)
python Evaluate.py

# Only (re)draw the plot from cached state
python Evaluate.py --plot-only

# Only (re)generate the txt report from cached state
python Evaluate.py --report-only
"""
import os
import argparse
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from river import metrics

# ——— CONFIG ——————————————————————————————————————————————
MODEL_PATH          = r".\output\SelfLabel2\23_SelfLabel_24shuffled.joblib"
EVAL_DATA_FOLDER    = r"d:\Downsampling\CICIOMT2024\just_shuffling\test"

'''
SELECTED_FEATURES   = ['Number', 'ack_flag_number', 'HTTPS', 'Tot size',
                    'Header_Length', 'IAT', 'Rate', 'AVG', 'ack_count', 
                    'Variance']
'''

'''
SELECTED_FEATURES   = ['Header_Length','IRC','UDP','DHCP','ARP',
                       'ICMP','IGMP','IPv','LLC','Tot sum']
'''

SELECTED_FEATURES = ['avg', 'header_length', 'std', 'iat', 'ack_flag_number']

REPORT_PATH         = r".\output\SelfLabel2\CrossValidate\23_SelfLabel_24shuffled_CV24.txt"
PLOT_PATH           = r".\output\Adapt\Cross Validate\23_SelfLabel_24shuffled_CV24.png"
PROCESSED_LOG       = r".\output\Adapt\Cross Validate\23_SelfLabel_24shuffled_CV24.txt"
STATE_PATH          = r".\output\Adapt\Cross Validate\23_SelfLabel_24shuffled_CV24.joblib"
# ————————————————————————————————————————————————————————

# ---------- helper functions --------------------------------
def get_processed_files() -> set[str]:
    if not os.path.exists(PROCESSED_LOG):
        return set()
    with open(PROCESSED_LOG, "r") as f:
        return {line.strip() for line in f}

def mark_as_processed(filename: str) -> None:
    with open(PROCESSED_LOG, "a") as f:
        f.write(filename + "\n")

# ---------- Evaluator class ---------------------------------
class Evaluator:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ model not found: {model_path}")
        self.model, self.scaler = joblib.load(model_path)
        print(f"✅ loaded model & scaler from {model_path}")

        # fresh metrics (will be overwritten if state exists)
        self.accuracy      = metrics.Accuracy()
        self.precision     = metrics.Precision()
        self.recall        = metrics.Recall()
        self.f1            = metrics.F1()
        self.conf_matrix   = metrics.ConfusionMatrix()
        self.data_points   : list[int]   = []
        self.accuracies    : list[float] = []

        # restore old state if present
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "rb") as f:
                state = joblib.load(f)
            self.accuracy     = state["accuracy"]
            self.precision    = state["precision"]
            self.recall       = state["recall"]
            self.f1           = state["f1"]
            self.conf_matrix  = state["conf_matrix"]
            self.data_points  = state["data_points"]
            self.accuracies   = state["accuracies"]
            print("📦 previous evaluation state loaded.")

    # ——— I/O helpers ————————————————————————————————
    def _save_state(self):
        state = dict(accuracy=self.accuracy, precision=self.precision,
                     recall=self.recall,   f1=self.f1,
                     conf_matrix=self.conf_matrix,
                     data_points=self.data_points,
                     accuracies=self.accuracies)
        with open(STATE_PATH, "wb") as f:
            joblib.dump(state, f)
        print(f"💾 state saved → {STATE_PATH}")

    # ——— evaluation on one CSV ——————————
    def evaluate_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)

        # Normalize column names to lowercase with underscores
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        if 'label' not in df.columns:
            print(f"⚠️  skipped (no Label): {csv_path}")
            return
        df['label'] = df['label'].str.lower().eq('benign').astype(int).rsub(1)
        df = df[SELECTED_FEATURES + ['label']].dropna()

        for _, row in df.iterrows():
            x_raw = row.drop('label').to_dict()
            y_true = row['label']
            x = self.scaler.transform_one(x_raw)
            y_pred = self.model.predict_one(x)

            if y_pred is not None:
                self.accuracy.update(y_true, y_pred)
                self.precision.update(y_true, y_pred)
                self.recall.update(y_true, y_pred)
                self.f1.update(y_true, y_pred)
                self.conf_matrix.update(y_true, y_pred)

                self.data_points.append(len(self.data_points))
                self.accuracies.append(self.accuracy.get())

    # ——— report / plot ——————————————————————————
    def write_report(self, path: str = REPORT_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("=== Evaluation Metrics ===\n")
            f.write(f"Accuracy : {self.accuracy.get():.4f}\n")
            f.write(f"Precision: {self.precision.get():.4f}\n")
            f.write(f"Recall   : {self.recall.get():.4f}\n")
            f.write(f"F1 Score : {self.f1.get():.4f}\n\n")

            f.write("=== Confusion Matrix ===\n")
            f.write(f"{self.conf_matrix}\n\n")

            # extended rates
            try:
                cm = self.conf_matrix
                TP,TN = cm[1][1], cm[0][0]
                FP,FN = cm[0][1], cm[1][0]
                TPR = TP/(TP+FN) if (TP+FN) else 0
                TNR = TN/(TN+FP) if (TN+FP) else 0
                FPR = FP/(FP+TN) if (FP+TN) else 0
                FNR = FN/(FN+TP) if (FN+TP) else 0
                f.write("=== Extended Metrics ===\n")
                f.write(f"TPR: {TPR:.4f}  TNR: {TNR:.4f}\n")
                f.write(f"FPR: {FPR:.4f}  FNR: {FNR:.4f}\n")
            except Exception as e:
                f.write(f"Error computing extended metrics: {e}\n")
        print(f"📄 report written → {path}")

    def plot(self, path: str = PLOT_PATH):
        if not self.accuracies:
            print("⚠️  no data to plot.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.figure(figsize=(10,5))
        plt.plot(self.data_points, self.accuracies,
                 label="Accuracy over time", color='green')
        plt.xlabel("samples seen")
        plt.ylabel("accuracy")
        plt.title("Model evaluation accuracy trend")
        plt.ylim(0.75, 1.0)          # ← feel free to tweak & rerun
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"📊 plot written → {path}")

# ---------- main driver ------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate model or regenerate report/plot.")
    parser.add_argument("--plot-only",   action="store_true",
                        help="Regenerate only the PNG plot from cached state.")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate only the TXT report from cached state.")
    args = parser.parse_args()

    # Decide what to do
    run_evaluation = not (args.plot_only or args.report_only)

    evaluator = Evaluator(MODEL_PATH)

    if run_evaluation:
        processed = get_processed_files()
        for fname in sorted(os.listdir(EVAL_DATA_FOLDER)):
            if fname.endswith(".csv") and fname not in processed:
                evaluator.evaluate_csv(os.path.join(EVAL_DATA_FOLDER, fname))
                mark_as_processed(fname)
        evaluator._save_state()

    # Generate outputs as requested
    if not args.plot_only:   # report or full eval
        evaluator.write_report()
    if not args.report_only: # plot or full eval
        evaluator.plot()

if __name__ == "__main__":
    main()
