"""
Create a clean, balanced dataset with equal real and fake news
"""

import pandas as pd
import os

def create_clean_dataset():
    """Create a clean, balanced dataset"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    print("=" * 80)
    print("CREATING CLEAN BALANCED DATASET")
    print("=" * 80)
    
    # Load current dataset
    current_path = os.path.join(data_dir, 'balanced_training_data.csv')
    df = pd.read_csv(current_path)
    
    print(f"\nCurrent dataset:")
    print(f"  Total: {len(df)}")
    print(f"  Real: {(df['label']=='real').sum()}")
    print(f"  Fake: {(df['label']=='fake').sum()}")
    
    # Separate real and fake
    real_df = df[df['label'] == 'real']
    fake_df = df[df['label'] == 'fake']
    
    # Balance: take equal amounts
    n_samples = min(len(real_df), len(fake_df))
    
    # If we have very few real news, take more fake but still balanced
    if n_samples < 100:
        print(f"\n⚠ Warning: Only {n_samples} samples per class available")
        print(f"  Recommendation: Download True.csv from Kaggle for better results")
    
    print(f"\n✓ Creating balanced dataset with {n_samples} samples per class...")
    
    # Sample equally
    real_sample = real_df.sample(n=n_samples, random_state=42)
    fake_sample = fake_df.sample(n=n_samples, random_state=42)
    
    # Combine
    balanced_df = pd.concat([real_sample, fake_sample], ignore_index=True)
    
    # Clean text: remove extra spaces, newlines
    print("\n✓ Cleaning text...")
    balanced_df['text'] = balanced_df['text'].str.strip()
    balanced_df['text'] = balanced_df['text'].str.replace(r'\s+', ' ', regex=True)
    
    # Remove any duplicates
    initial_len = len(balanced_df)
    balanced_df = balanced_df.drop_duplicates(subset=['text'])
    if len(balanced_df) < initial_len:
        print(f"  - Removed {initial_len - len(balanced_df)} duplicates")
    
    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    output_path = os.path.join(data_dir, 'clean_balanced_dataset.csv')
    balanced_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Clean balanced dataset created!")
    print(f"  Total: {len(balanced_df)}")
    print(f"  Real: {(balanced_df['label']=='real').sum()}")
    print(f"  Fake: {(balanced_df['label']=='fake').sum()}")
    print(f"  Saved to: {output_path}")
    
    # Show sample
    print(f"\n📋 Sample of data:")
    print("-" * 80)
    for i in range(min(3, len(balanced_df))):
        row = balanced_df.iloc[i]
        text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
        print(f"{i+1}. [{row['label'].upper()}] {text_preview}")
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"\nTo train with this clean dataset:")
    print(f"  python src/quick_train_improved.py --data data/clean_balanced_dataset.csv")
    print("=" * 80)
    
    return output_path

if __name__ == "__main__":
    create_clean_dataset()
