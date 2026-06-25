from pathlib import Path
import polars as pl
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def split_parquet_data():

    # Configuration constants
    INPUT_FILE = "data/vllm_processed/labeled_subset.pqt"
    OUTPUT_DIR = "data/processed"
    SEED = 42

    # Create paths and output dir
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Streaming dataset via Polars from {input_path}...")
    lazy_df = pl.scan_parquet(input_path)
    
    # Filter out empty records
    lazy_df = lazy_df.filter(
        pl.col("message_text").is_not_null() & 
        (pl.col("message_text").str.strip_chars() != "")
    )
    df = lazy_df.collect()
    total_rows = len(df)
    print(f"Dataset loaded. Total clean rows: {total_rows}")

    # Define your multi-label target columns to guide the stratification
    label_columns = [
        'elite_vs_mass_conflict',
        'in_group_vs_out_group_exclusion',
        'institutional_knowledge_denial',
        'societal_moral_regression',
        'imminent_acute_crisis_panic',
        'systemic_sovereignty_revival'
        # Once you generate your child narrative columns, append their names here!
    ]
    
    # Create the temporary binary matrix needed for stratification calculation
    y_stratify = (df[label_columns].to_numpy() >= 2).astype(int)
    X_indices = np.arange(total_rows).reshape(-1, 1)

    print("Executing Optimized Multilabel Stratification")
    
    # Separate Test Set (10%)
    msss_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    train_val_idx, test_idx = next(msss_test.split(X_indices, y_stratify))
    
    # Isolate temporary datasets based on index positions
    df_train_val = df[train_val_idx]
    y_train_val_stratify = y_stratify[train_val_idx]
    X_train_val_indices = np.arange(len(df_train_val)).reshape(-1, 1)

    # Split remaining 90% into Train (80% total) and Val (10% total)
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1111, random_state=SEED)
    train_idx, val_idx = next(msss_val.split(X_train_val_indices, y_train_val_stratify))

    # Slice out final raw data splits
    train_df = df_train_val[train_idx]
    val_df = df_train_val[val_idx]
    test_df = df[test_idx]

    print("\n=== Balanced Split Distribution Results ===")
    print(f"Train set: {train_df.height} rows ({train_df.height/total_rows*100:.1f}%)")
    print(f"Val set:   {val_df.height} rows ({val_df.height/total_rows*100:.1f}%)")
    print(f"Test set:  {test_df.height} rows ({test_df.height/total_rows*100:.1f}%)")

    # Save to disk
    train_df.write_parquet(output_path / "train_split.pqt")
    val_df.write_parquet(output_path / "val_split.pqt")
    test_df.write_parquet(output_path / "test_split.pqt")
    print(f"\nSaved stratified splits safely to: {output_path}")

if __name__ == "__main__":
    split_parquet_data()