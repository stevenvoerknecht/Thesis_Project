import pandas as pd
import matplotlib.pyplot as plt
import polars as pl

def load_data_pqt(file_path):
    """Load data from a Parquet (.pqt) file."""
    try:
        data = pl.scan_parquet(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading Parquet data: {e}")
        return None

if __name__ == "__main__":
    # Load the data
    file_path = "data/vllm_processed/labeled_subset_v4_2.pqt"
    data = load_data_pqt(file_path)

    # Do not truncate text inside columns
    pl.Config.set_fmt_str_lengths(100)  # Adjust 1000 to the max character length you want to see

    # Optional: Show more rows or columns if needed
    pl.Config.set_tbl_rows(20)

    if data is not None:
        # Display basic information about the dataset
        print("\nFirst 5 rows of the dataset:")
        print(data.head(5).collect())

        # View the column names and data types
        print("\nColumn Names and Data Types:")
        print(data.schema)

        # View a few of the messages 
        print("\nSample Messages:")
        print(data.select(pl.col("message_text")).head(10).collect())

        print("\nLast message")
        last_message_df = data.select(pl.col("message_text")).tail(1).collect()
        print(data.select(pl.col("message_id")).tail(1).collect())
        print(last_message_df)
        print(last_message_df.item())