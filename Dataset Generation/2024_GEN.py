import os
import pandas as pd
from sklearn.utils import resample

# === CONFIG ===
dataset_root = r"D:\Downsampling\CICIOMT2024\labelled"
train_dir = os.path.join(dataset_root, "train")  # Pre-split train files
test_dir = os.path.join(dataset_root, "test")    # Pre-split test files
output_dir = r"D:\Downsampling\CICIOMT2024\Down"
os.makedirs(output_dir, exist_ok=True)

random_state = 42

# Sampling rules
MALICIOUS_SAMPLE_RATE = 0.15
MIN_MALICIOUS_SAMPLES = 100
MAX_MALICIOUS_SAMPLES = 50000

# === PROCESSING FUNCTION ===
def process_split(split_dir, split_name):
    print(f"\n⏳ Processing {split_name} files...")
    all_files = [f for f in os.listdir(split_dir) if f.endswith('.csv')]

    malicious_dfs = []
    benign_dfs = []

    for f in all_files:
        fpath = os.path.join(split_dir, f)
        try:
            df = pd.read_csv(fpath)

            # Clean column names
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

            # Remove any existing 'label' column to avoid duplication
            df = df.drop(columns=["label"], errors="ignore")

            if "benign" in f.lower():
                df['label'] = 'Benign'
                benign_dfs.append(df)
            else:
                sample_size = min(
                    MAX_MALICIOUS_SAMPLES,
                    max(MIN_MALICIOUS_SAMPLES, int(len(df) * MALICIOUS_SAMPLE_RATE))
                )
                df = df.sample(n=sample_size, random_state=random_state)
                df['label'] = 'Malicious'
                malicious_dfs.append(df)

            print(f"✅ Processed {f} ({len(df)} rows)")

        except Exception as e:
            print(f"⚠ Failed to process {f}: {str(e)[:100]}")

    # Combine
    malicious_combined = pd.concat(malicious_dfs, ignore_index=True) if malicious_dfs else pd.DataFrame()
    benign_combined = pd.concat(benign_dfs, ignore_index=True) if benign_dfs else pd.DataFrame()

    print(f"\nOriginal {split_name} counts:")
    print(f"Malicious: {len(malicious_combined)}")
    print(f"Benign: {len(benign_combined)}")

    if len(malicious_combined) > 0 and len(benign_combined) > 0:
        # Balance both classes
        if len(benign_combined) < len(malicious_combined):
            benign_balanced = resample(
                benign_combined,
                replace=True,
                n_samples=len(malicious_combined),
                random_state=random_state
            )
            balanced_df = pd.concat([malicious_combined, benign_balanced])
        else:
            malicious_balanced = resample(
                malicious_combined,
                replace=True,
                n_samples=len(benign_combined),
                random_state=random_state
            )
            balanced_df = pd.concat([malicious_balanced, benign_combined])

        # Shuffle and remove duplicate columns
        balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        balanced_df = balanced_df.loc[:, ~balanced_df.columns.duplicated()]

        # Save to CSV
        output_path = os.path.join(output_dir, f"{split_name}_dataset.csv")
        balanced_df.to_csv(output_path, index=False)

        print(f"\n✅ Balanced {split_name} dataset saved ({len(balanced_df)} rows)")
        print("Class distribution:")
        print(balanced_df['label'].value_counts(normalize=True))
        return balanced_df
    else:
        print(f"⚠ Could not balance {split_name} split due to missing classes")
        return None

# === MAIN EXECUTION ===
if __name__ == "__main__":
    train_df = process_split(train_dir, "train")
    test_df = process_split(test_dir, "test")

    if train_df is not None and test_df is not None:
        print("\n🎉 Successfully processed both splits!")
        print(f"Final train samples: {len(train_df)}")
        print(f"Final test samples: {len(test_df)}")
    else:
        print("\n⚠ Processing incomplete - check error messages above")
