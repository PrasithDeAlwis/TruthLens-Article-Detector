"""
Clean and preprocess dataset - remove punctuation, numbers, stopwords
Make text clean for better model accuracy
"""

import pandas as pd
import re
import string
import os
from preprocess import TextPreprocessor

def clean_dataset_thoroughly():
    """Clean dataset by removing punctuation, numbers, extra spaces"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    print("=" * 80)
    print("CLEANING DATASET - REMOVING PUNCTUATION, NUMBERS, STOPWORDS")
    print("=" * 80)
    
    # Load dataset
    input_path = os.path.join(data_dir, 'balanced_training_data.csv')
    print(f"\n[1/5] Loading dataset...")
    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df)} articles")
    print(f"  - Real: {(df['label']=='real').sum()}")
    print(f"  - Fake: {(df['label']=='fake').sum()}")
    
    # Balance the dataset first
    print(f"\n[2/5] Balancing dataset...")
    real_df = df[df['label'] == 'real']
    fake_df = df[df['label'] == 'fake']
    
    n_samples = min(len(real_df), len(fake_df))
    print(f"  - Taking {n_samples} samples from each class")
    
    real_sample = real_df.sample(n=n_samples, random_state=42)
    fake_sample = fake_df.sample(n=n_samples, random_state=42)
    
    df = pd.concat([real_sample, fake_sample], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✓ Balanced dataset: {len(df)} articles ({n_samples} real + {n_samples} fake)")
    
    # Clean text thoroughly
    print(f"\n[3/5] Cleaning text...")
    print("  - Removing URLs, emails, HTML tags")
    print("  - Removing numbers and punctuation")
    print("  - Removing stopwords")
    print("  - Converting to lowercase")
    print("  - Lemmatizing words")
    
    preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
    
    cleaned_texts = []
    for i, text in enumerate(df['text']):
        if (i + 1) % 100 == 0:
            print(f"  - Progress: {i+1}/{len(df)}")
        
        cleaned = preprocessor.preprocess(text)
        cleaned_texts.append(cleaned)
    
    df['cleaned_text'] = cleaned_texts
    
    # Remove empty or very short texts
    print(f"\n[4/5] Removing empty/short texts...")
    df['text_length'] = df['cleaned_text'].str.split().str.len()
    df = df[df['text_length'] >= 10]  # At least 10 words
    df = df.drop('text_length', axis=1)
    
    print(f"✓ Removed short texts, remaining: {len(df)} articles")
    
    # Final dataset
    final_df = df[['text', 'cleaned_text', 'label']]
    
    # Save
    print(f"\n[5/5] Saving cleaned dataset...")
    output_path = os.path.join(data_dir, 'clean_preprocessed_dataset.csv')
    final_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved to: {output_path}")
    
    # Show examples
    print(f"\n" + "=" * 80)
    print("EXAMPLES OF CLEANED TEXT")
    print("=" * 80)
    
    for i in range(min(3, len(final_df))):
        row = final_df.iloc[i]
        print(f"\n{i+1}. Label: {row['label'].upper()}")
        print(f"   Original: {row['text'][:150]}...")
        print(f"   Cleaned:  {row['cleaned_text'][:150]}...")
    
    # Statistics
    print(f"\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"  Total articles: {len(final_df)}")
    print(f"  Real news: {(final_df['label']=='real').sum()}")
    print(f"  Fake news: {(final_df['label']=='fake').sum()}")
    print(f"  Average words (original): {final_df['text'].str.split().str.len().mean():.0f}")
    print(f"  Average words (cleaned): {final_df['cleaned_text'].str.split().str.len().mean():.0f}")
    
    print(f"\n" + "=" * 80)
    print("DONE! Dataset is now clean and ready for training")
    print("=" * 80)
    print(f"\nNext step - Train with cleaned data:")
    print(f"  python src/quick_train_improved.py --data data/clean_preprocessed_dataset.csv")
    print("=" * 80)
    
    return output_path

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    clean_dataset_thoroughly()
