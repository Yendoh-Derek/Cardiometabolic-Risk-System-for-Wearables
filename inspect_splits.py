import pandas as pd

# Load the parquet files created by Phase 5A
train_df = pd.read_parquet('data/processed/ssl_pretraining_data.parquet')
val_df = pd.read_parquet('data/processed/ssl_validation_data.parquet')

print("Train metadata:")
print("  Shape:", train_df.shape)
print("  Columns:", list(train_df.columns))
if 'subject_id' in train_df.columns:
    print("  Unique subjects:", train_df['subject_id'].nunique())
    print("  Subject IDs sample:", sorted(train_df['subject_id'].unique())[:10])
    
print("\nVal metadata:")
print("  Shape:", val_df.shape)
if 'subject_id' in val_df.columns:
    print("  Unique subjects:", val_df['subject_id'].nunique())
    print("  Subject IDs:", sorted(val_df['subject_id'].unique()))
