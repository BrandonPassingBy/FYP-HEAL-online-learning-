import os
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# === CONFIG ===
folder_path = r"D:\Downsampling\CICIOT2023\labelled"
output_dir = r"D:\Downsampling\CICIOT2023\Down"
os.makedirs(output_dir, exist_ok=True)

random_state = 42
test_size = 0.2

MALICIOUS_SAMPLE_RATE = 0.20
MIN_MALICIOUS_SAMPLES = 100
MAX_MALICIOUS_SAMPLES = 50000

# === STEP 1: Clean Column Names ===
def clean_columns(df):
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df

# === STEP 2: Sample File ===
def sample_file(fpath, label):
    try:
        print(f"Processing {os.path.basename(fpath)}...")
        df = pd.read_csv(fpath)

        # Drop completely empty rows and columns
        df.dropna(axis=1, how='all', inplace=True)
        df.dropna(axis=0, how='all', inplace=True)

        df = clean_columns(df)

        # Remove exact duplicate rows
        df.drop_duplicates(inplace=True)

        if label == 'malicious':
            sample_size = min(
                MAX_MALICIOUS_SAMPLES,
                max(MIN_MALICIOUS_SAMPLES, int(len(df) * MALICIOUS_SAMPLE_RATE))
            )
            df = df.sample(n=sample_size, random_state=random_state)

        df['label'] = label
        print(f"✅ {os.path.basename(fpath)} processed: {len(df)} rows")
        return df
    except Exception as e:
        print(f"⚠ Error processing {os.path.basename(fpath)}: {str(e)}")
        return None

# === STEP 3: Load & Sample Files ===
print("🔍 Scanning directory...")
all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
print(f"Found {len(all_files)} CSV files.")

malicious_dfs = []
benign_dfs = []

for f in all_files:
    fpath = os.path.join(folder_path, f)
    if "benign" in f.lower():
        df = sample_file(fpath, 'benign')
        if df is not None:
            benign_dfs.append(df)
    else:
        df = sample_file(fpath, 'malicious')
        if df is not None:
            malicious_dfs.append(df)

# === STEP 4: Combine and Balance ===
try:
    malicious_combined = pd.concat(malicious_dfs, ignore_index=True) if malicious_dfs else pd.DataFrame()
    benign_combined = pd.concat(benign_dfs, ignore_index=True) if benign_dfs else pd.DataFrame()

    print(f"\n📊 Total Malicious: {len(malicious_combined)} | Benign: {len(benign_combined)}")

    # Drop duplicates across all
    malicious_combined.drop_duplicates(inplace=True)
    benign_combined.drop_duplicates(inplace=True)

    if not malicious_combined.empty and not benign_combined.empty:
        # Balance the dataset
        if len(benign_combined) < len(malicious_combined):
            benign_balanced = resample(
                benign_combined,
                replace=True,
                n_samples=len(malicious_combined),
                random_state=random_state
            )
        else:
            benign_balanced = benign_combined

        final_df = pd.concat([malicious_combined, benign_balanced])
        final_df = final_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Train-test split
        train_df, test_df = train_test_split(
            final_df,
            test_size=test_size,
            random_state=random_state,
            stratify=final_df['label']
        )

        # === Save ===
        train_df.to_csv(os.path.join(output_dir, "train_dataset.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test_dataset.csv"), index=False)

        # === Summary ===
        print(f"\n✅ Final Dataset: {len(final_df):,} rows")
        print(f"📂 Train: {len(train_df):,} | Test: {len(test_df):,}")
        print("\nClass Balance in Final Dataset:")
        print(final_df['label'].value_counts(normalize=True).round(3))
    else:
        print("\n⚠ Dataset balancing skipped because:")
        if malicious_combined.empty:
            print("- No malicious samples found.")
        if benign_combined.empty:
            print("- No benign samples found.")

except Exception as e:
    print(f"\n❌ Critical error: {str(e)}")
    raise
