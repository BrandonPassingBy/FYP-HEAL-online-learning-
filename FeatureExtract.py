import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import gc
import psutil
import sys

# Configuration
LABELLED_FOLDER = "/home/b/Dataset/dataset/CICIOMT2024/pro/test2"
PROGRESS_LOG = "./_feature_progress_log.txt"
RESULTS_FILE = "./top_safe_features.txt"
CHECKPOINT_FILE = "./_feature_checkpoint.pkl"

# Only safe features to use (customize as needed)
SAFE_FEATURES = [
    'Protocol Type', 'Rate', 'Duration', 'Header_Length',
    'ack_flag_number', 'syn_flag_number', 'rst_flag_number',
    'ack_count', 'syn_count', 'HTTP', 'TCP', 'UDP', 'ICMP',
    'AVG', 'Std', 'IAT'
]

# Memory safety threshold (in MB)
MEMORY_THRESHOLD = 500  # Start processing when this much memory is available

def get_available_memory():
    """Returns available memory in MB"""
    return psutil.virtual_memory().available / (1024 * 1024)

def process_in_batches(file_list, batch_size=5):
    """Process files in batches to manage memory"""
    for i in range(0, len(file_list), batch_size):
        batch = file_list[i:i + batch_size]
        batch_dfs = []
        
        for filename in batch:
            try:
                path = os.path.join(LABELLED_FOLDER, filename)
                df = pd.read_csv(path, usecols=lambda c: c.strip() in SAFE_FEATURES + ['Label'])
                
                df.columns = df.columns.str.strip()
                
                # Keep the original Label column (no binary conversion)
                if df['Label'].isnull().any():
                    print(f"⚠️ Skipping {filename}: contains NaNs in Label")
                    continue

                # Reorder columns to safe features + Label
                df = df[SAFE_FEATURES + ['Label']]

                if df.isnull().values.any():
                    print(f"⚠️ Skipping {filename}: contains NaNs in features")
                    continue

                batch_dfs.append(df)
                print(f"✅ Loaded: {filename} → {df.shape[0]} rows")

                with open(PROGRESS_LOG, "a") as f:
                    f.write(filename + "\n")

            except Exception as e:
                print(f"❌ Error in {filename}: {str(e)[:100]}")  # Truncate long error messages

        if batch_dfs:
            yield pd.concat(batch_dfs, ignore_index=True)
        
        # Explicit cleanup
        del batch_dfs
        gc.collect()

def load_checkpoint():
    """Load checkpoint if exists"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            checkpoint = pd.read_pickle(CHECKPOINT_FILE)
            print(f"🔁 Resuming from checkpoint with {checkpoint.shape[0]} rows")
            return checkpoint
        except:
            print("⚠️ Corrupted checkpoint, starting fresh")
            return None
    return None

def save_checkpoint(df):
    """Save current progress"""
    df.to_pickle(CHECKPOINT_FILE)
    print(f"💾 Checkpoint saved with {df.shape[0]} rows")

def main():
    # Keep track of processed files
    if os.path.exists(PROGRESS_LOG):
        with open(PROGRESS_LOG, "r") as f:
            processed = set(line.strip() for line in f)
    else:
        processed = set()

    # Get files to process
    files_to_process = [
        f for f in os.listdir(LABELLED_FOLDER) 
        if f.endswith(".csv") and f not in processed
    ]
    
    if not files_to_process and not os.path.exists(CHECKPOINT_FILE):
        print("✅ All files already processed or no new files found")
        if os.path.exists(RESULTS_FILE):
            print(f"Results already exist at {RESULTS_FILE}")
            return
        else:
            print("❌ No files to process and no results found")
            return

    # Initialize combined_df from checkpoint or empty
    combined_df = load_checkpoint() or pd.DataFrame()

    # Process files in batches
    try:
        for batch_df in process_in_batches(files_to_process, batch_size=3):  # Smaller batch size for memory safety
            # Check memory before processing
            if get_available_memory() < MEMORY_THRESHOLD:
                print(f"⚠️ Low memory ({get_available_memory():.1f}MB available), saving checkpoint and exiting")
                save_checkpoint(combined_df)
                sys.exit(0)  # Exit cleanly to allow resuming
            
            # Combine with existing data
            combined_df = pd.concat([combined_df, batch_df], ignore_index=True)
            
            # Save checkpoint periodically
            if len(combined_df) % 10000 == 0:  # Adjust based on your dataset size
                save_checkpoint(combined_df)
                
            # Clean up
            del batch_df
            gc.collect()

        # Final feature selection if we have data
        if not combined_df.empty:
            print(f"\n🧮 Final dataset shape: {combined_df.shape}")

            # Prepare features and target label
            X = combined_df[SAFE_FEATURES]
            y = combined_df['Label']

            print("🧪 Training RandomForest (this may take a while)...")
            
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                verbose=1,
                max_samples=0.5 if len(combined_df) > 100000 else None  # Subsample if large dataset
            )
            
            rf.fit(X, y)

            # Get feature importances
            importances = rf.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values(by='importance', ascending=False)

            top_features_df = feature_importance_df.head(10)
            selected_features = top_features_df['feature'].tolist()

            print(f"\n🏆 Top 10 Features:")
            print(selected_features)

            # Save result
            top_features_df.to_csv(RESULTS_FILE, sep='\t', index=False)
            print(f"💾 Saved top features to: {RESULTS_FILE}")

            # Clean up checkpoint
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
        else:
            print("❌ No data processed")

    except (MemoryError, KeyboardInterrupt) as e:
        print(f"\n⚠️ Interrupted by {'user' if isinstance(e, KeyboardInterrupt) else 'memory error'}")
        if not combined_df.empty:
            save_checkpoint(combined_df)
            print("Checkpoint saved. You can resume later.")
        sys.exit(1)

if __name__ == "__main__":
    main()
