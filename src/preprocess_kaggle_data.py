"""
Preprocess Kaggle Fake News Dataset
This script loads the Fake.csv file, cleans and preprocesses it, and saves it for training
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import TextPreprocessor

def preprocess_kaggle_fake_news():
    """
    Preprocess the Fake.csv dataset from Kaggle
    """
    print("=" * 80)
    print("PREPROCESSING KAGGLE FAKE NEWS DATASET")
    print("=" * 80)
    
    # Load the dataset
    print("\n[1/6] Loading Fake.csv dataset...")
    
    # Get the correct path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    fake_csv_path = os.path.join(project_root, 'Fake.csv')
    
    try:
        df = pd.read_csv(fake_csv_path, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(fake_csv_path, encoding='latin-1')
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print(f"Looking for file at: {fake_csv_path}")
            return
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Columns: {list(df.columns)}")
    
    # Explore the dataset structure
    print("\n[2/6] Analyzing dataset structure...")
    print(f"  - First few rows:")
    print(df.head())
    print(f"\n  - Dataset info:")
    print(df.info())
    print(f"\n  - Missing values:")
    print(df.isnull().sum())
    
    # Identify text and label columns
    print("\n[3/6] Identifying text and label columns...")
    
    # Common column names in Kaggle fake news datasets
    text_column = None
    label_column = None
    
    # Check for text column
    for col in ['text', 'content', 'article', 'body', 'news', 'title']:
        if col in df.columns.str.lower():
            text_column = df.columns[df.columns.str.lower() == col][0]
            break
    
    # If not found, combine title and text
    if text_column is None:
        if 'title' in df.columns and 'text' in df.columns:
            print("  - Combining 'title' and 'text' columns...")
            df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
            text_column = 'combined_text'
        elif 'title' in df.columns:
            text_column = 'title'
        elif 'text' in df.columns:
            text_column = 'text'
        else:
            # Use the column with longest text
            text_column = df.select_dtypes(include=['object']).apply(lambda x: x.str.len().mean()).idxmax()
    
    # Check for label column
    for col in ['label', 'class', 'category', 'type', 'fake']:
        if col in df.columns.str.lower():
            label_column = df.columns[df.columns.str.lower() == col][0]
            break
    
    if label_column is None:
        print("  ! No label column found. Checking for separate files or creating labels...")
        # The Fake.csv might be all fake news, we'll label it as 'fake'
        df['label'] = 'fake'
        label_column = 'label'
    
    print(f"✓ Identified columns:")
    print(f"  - Text column: {text_column}")
    print(f"  - Label column: {label_column}")
    
    # Clean the dataset
    print("\n[4/6] Cleaning dataset...")
    
    # Remove rows with missing text
    initial_rows = len(df)
    df = df.dropna(subset=[text_column])
    df = df[df[text_column].astype(str).str.strip() != '']
    
    # Remove duplicates
    df = df.drop_duplicates(subset=[text_column])
    
    print(f"✓ Cleaning complete:")
    print(f"  - Rows removed: {initial_rows - len(df)}")
    print(f"  - Remaining rows: {len(df)}")
    
    # Standardize labels
    print("\n[5/6] Standardizing labels...")
    print(f"  - Original label distribution:")
    print(df[label_column].value_counts())
    
    # Standardize to 'real' and 'fake'
    df['label'] = df[label_column].astype(str).str.lower().str.strip()
    
    # Map various label formats to 'real' and 'fake'
    label_mapping = {
        '1': 'fake', '0': 'real',
        'true': 'fake', 'false': 'real',
        'fake': 'fake', 'real': 'real',
        'unreliable': 'fake', 'reliable': 'real',
        'fake news': 'fake', 'real news': 'real'
    }
    
    df['label'] = df['label'].map(lambda x: label_mapping.get(x, x))
    
    # If all entries are 'fake', we need to find a 'True.csv' or mark them accordingly
    if df['label'].nunique() == 1:
        print(f"  ! All entries have the same label: {df['label'].iloc[0]}")
        print(f"  - Checking for True.csv or Real.csv...")
        
        # Try to load a corresponding real news file
        real_df = None
        for real_file in ['True.csv', 'Real.csv', 'real.csv', 'true.csv']:
            real_path = os.path.join(project_root, real_file)
            if os.path.exists(real_path):
                print(f"  ✓ Found {real_file}, loading...")
                try:
                    real_df = pd.read_csv(real_path, encoding='utf-8')
                except:
                    real_df = pd.read_csv(real_path, encoding='latin-1')
                break
        
        if real_df is not None:
            # Process real news the same way
            print(f"  - Processing real news dataset...")
            if text_column in real_df.columns or 'text' in real_df.columns:
                if 'title' in real_df.columns and 'text' in real_df.columns:
                    real_df['combined_text'] = real_df['title'].fillna('') + ' ' + real_df['text'].fillna('')
                    real_text_col = 'combined_text'
                elif text_column in real_df.columns:
                    real_text_col = text_column
                else:
                    real_text_col = 'text'
                
                real_df = real_df.dropna(subset=[real_text_col])
                real_df = real_df[real_df[real_text_col].astype(str).str.strip() != '']
                real_df = real_df.drop_duplicates(subset=[real_text_col])
                
                # Create standardized dataframe
                real_df_clean = pd.DataFrame({
                    'text': real_df[real_text_col],
                    'label': 'real'
                })
                
                # Combine fake and real news
                df_clean = pd.DataFrame({
                    'text': df[text_column],
                    'label': 'fake'
                })
                
                df = pd.concat([df_clean, real_df_clean], ignore_index=True)
                print(f"  ✓ Combined datasets: {len(df)} total rows")
            else:
                df['label'] = 'fake'
        else:
            print(f"  ! No corresponding real news file found.")
            print(f"  ! Dataset will be unbalanced with only fake news.")
    
    # Create final clean dataframe
    df_final = pd.DataFrame({
        'text': df[text_column] if text_column in df.columns else df['text'],
        'label': df['label']
    })
    
    # Remove any remaining issues
    df_final = df_final.dropna()
    df_final = df_final.reset_index(drop=True)
    
    print(f"\n✓ Final label distribution:")
    print(df_final['label'].value_counts())
    print(f"\n✓ Final dataset shape: {df_final.shape}")
    
    # Preprocess text
    print("\n[6/6] Preprocessing text...")
    print("  - Initializing text preprocessor...")
    preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
    
    print("  - Cleaning and preprocessing text (this may take a while)...")
    
    # Process in chunks to show progress
    chunk_size = 1000
    processed_texts = []
    
    for i in range(0, len(df_final), chunk_size):
        chunk = df_final['text'].iloc[i:i+chunk_size]
        processed_chunk = chunk.apply(preprocessor.preprocess)
        processed_texts.extend(processed_chunk.tolist())
        
        progress = min(i + chunk_size, len(df_final))
        print(f"  - Progress: {progress}/{len(df_final)} ({progress*100//len(df_final)}%)")
    
    df_final['processed_text'] = processed_texts
    
    # Remove rows where preprocessing resulted in empty text
    df_final = df_final[df_final['processed_text'].str.strip() != '']
    df_final = df_final.reset_index(drop=True)
    
    print(f"\n✓ Preprocessing complete!")
    print(f"  - Final dataset size: {len(df_final)} rows")
    
    # Save preprocessed data
    print("\n[7/7] Saving preprocessed data...")
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, 'preprocessed_kaggle_data.csv')
    df_final[['text', 'processed_text', 'label']].to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}")
    
    # Also create a sample for quick testing
    sample_size = min(5000, len(df_final))
    df_sample = df_final.sample(n=sample_size, random_state=42)
    sample_path = os.path.join(data_dir, 'kaggle_sample_5k.csv')
    df_sample[['text', 'processed_text', 'label']].to_csv(sample_path, index=False)
    print(f"✓ Created sample dataset (5000 rows): {sample_path}")
    
    # Display statistics
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\nDataset Statistics:")
    print(f"  - Total articles: {len(df_final)}")
    print(f"  - Fake news: {(df_final['label'] == 'fake').sum()}")
    print(f"  - Real news: {(df_final['label'] == 'real').sum()}")
    print(f"  - Average text length: {df_final['text'].str.len().mean():.0f} characters")
    print(f"  - Average processed text length: {df_final['processed_text'].str.len().mean():.0f} characters")
    print(f"\nFiles created:")
    print(f"  1. {output_path} - Full preprocessed dataset")
    print(f"  2. {sample_path} - Sample dataset for quick testing")
    print(f"\nNext steps:")
    print(f"  1. Train model: python src/train.py --data data/preprocessed_kaggle_data.csv")
    print(f"  2. Or use sample: python src/train.py --data data/kaggle_sample_5k.csv")
    print("=" * 80)


if __name__ == "__main__":
    preprocess_kaggle_fake_news()
