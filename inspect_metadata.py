import pandas as pd

# Check window metadata
windows_df = pd.read_parquet('data/processed/mimic_windows_metadata.parquet')
print("Windows metadata:")
print("  Shape:", windows_df.shape)
print("  subject_id type:", windows_df['subject_id'].dtype)
print("  subject_id sample:", windows_df['subject_id'].unique()[:5])
print()

# Check old train metadata
train_df = pd.read_parquet('data/processed/ssl_pretraining_data.parquet')
print("Old train metadata:")
print("  Shape:", train_df.shape)
print("  Columns:", list(train_df.columns))
if 'subject_id' in train_df.columns:
    print("  subject_id type:", train_df['subject_id'].dtype)
    print("  subject_id sample:", train_df['subject_id'].unique()[:5])
print()

# Check old val metadata
val_df = pd.read_parquet('data/processed/ssl_validation_data.parquet')
print("Old val metadata:")
print("  Shape:", val_df.shape)
if 'subject_id' in val_df.columns:
    print("  subject_id type:", val_df['subject_id'].dtype)
    print("  subject_id sample:", val_df['subject_id'].unique()[:5])
