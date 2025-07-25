import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# === CONFIG ===
dataset1_train_folder = r"D:\Downsampling\CICIOMT2024\labelled\train"
dataset2_folder       = r".\23_IND_Dataset\train"
output_dir            = r"C:\Users\brand\OneDrive\Documents\Sunway\FYP\Codes\downsample_files\3"
os.makedirs(output_dir, exist_ok=True)

random_state = 42
train_ratio = 0.7
test_ratio = 0.2
eval_ratio = 0.1

# Downsampling rules
MALICIOUS_SAMPLE_RATE = 0.15  # Keep 15% of malicious
BENIGN_SAMPLE_RATE = 1.0      # Keep all benign (adjust if needed)
MIN_SAMPLES = 15000           # Minimum samples per class

# === STEP 1: Load ALL CSV files with automatic label detection ===
def load_folder(folder_path, default_label=None):
    """Load all CSVs with auto malicious/benign detection from filenames."""
    all_dfs = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            filepath = os.path.join(folder_path, file)
            try:
                is_benign = "benign" in file.lower()
                label = "benign" if is_benign else "malicious"

                chunks = []
                for chunk in pd.read_csv(filepath, chunksize=100_000):
                    # Normalize column names
                    chunk.columns = (
                        chunk.columns
                        .str.strip()
                        .str.lower()
                        .str.replace(" ", "_")
                    )
                    # Remove duplicate columns
                    chunk = chunk.loc[:, ~chunk.columns.duplicated()]

                    # Assign label if missing
                    if 'label' not in chunk.columns:
                        chunk['label'] = label

                    # Downsample if malicious
                    if label == "malicious":
                        chunk = chunk.sample(frac=MALICIOUS_SAMPLE_RATE, random_state=random_state)
                    elif label == "benign":
                        chunk = chunk.sample(frac=BENIGN_SAMPLE_RATE, random_state=random_state)

                    chunks.append(chunk)

                if chunks:
                    combined = pd.concat(chunks, ignore_index=True)
                    all_dfs.append(combined)
                    print(f"✅ Loaded {file} as {label} ({len(combined)} rows)")
            except Exception as e:
                print(f"⚠ Error loading {file}: {str(e)}")
    
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

print("Loading Dataset1 train...")
train1 = load_folder(dataset1_train_folder)

print("Loading Dataset2...")
dataset2 = load_folder(dataset2_folder)

# === STEP 2: Combine All Data ===
combined = pd.concat([train1, dataset2]).reset_index(drop=True)

# === STEP 3: Balance Classes ===
def balance_classes(df, target_per_class=None):
    """Downsample or upsample to create balanced dataset (optional fixed size)."""
    class_counts = df['label'].value_counts()
    print("\n📊 Class distribution before balancing:")
    print(class_counts)

    if target_per_class is None:
        target_per_class = class_counts.min()

    balanced_dfs = []
    for label in class_counts.index:
        class_df = df[df['label'] == label]
        if len(class_df) > target_per_class:
            class_df = class_df.sample(n=target_per_class, random_state=random_state)
        elif len(class_df) < target_per_class:
            class_df = resample(
                class_df,
                replace=True,
                n_samples=target_per_class,
                random_state=random_state
            )
        balanced_dfs.append(class_df)

    balanced = pd.concat(balanced_dfs).reset_index(drop=True)
    print("\n✅ Class distribution after balancing:")
    print(balanced['label'].value_counts())
    return balanced

combined_balanced = balance_classes(combined, target_per_class=2000000)

# === STEP 4: 70-20-10 Stratified Split ===
train, temp = train_test_split(
    combined_balanced,
    train_size=train_ratio,
    stratify=combined_balanced['label'],
    random_state=random_state
)

test, eval_ = train_test_split(
    temp,
    test_size=eval_ratio/(test_ratio + eval_ratio),  # 10% of total = 33.3% of this subset
    stratify=temp['label'],
    random_state=random_state
)

# === STEP 5: Save Results ===
train.to_csv(os.path.join(output_dir, "train1.csv"), index=False)
test.to_csv(os.path.join(output_dir, "test1.csv"), index=False)
eval_.to_csv(os.path.join(output_dir, "eval1.csv"), index=False)

print(f"""
✅ Final Dataset Summary:
   Total Samples: {len(combined_balanced):,}
   Train Set    : {len(train):,} ({train_ratio*100}%)
   Test Set     : {len(test):,} ({test_ratio*100}%)
   Eval Set     : {len(eval_):,} ({eval_ratio*100}%)

📁 Saved to: {output_dir}
   ├── train1.csv
   ├── test1.csv
   └── eval1.csv
""")
