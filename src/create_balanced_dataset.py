"""
Create a balanced dataset by combining Fake.csv with real news
This script helps create a balanced dataset for better model training
"""

import pandas as pd
import os

def create_balanced_dataset():
    """
    Create a balanced dataset from the preprocessed fake news and real news
    """
    print("=" * 80)
    print("CREATING BALANCED DATASET")
    print("=" * 80)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # Check for True.csv
    true_csv_path = os.path.join(project_root, 'True.csv')
    
    if not os.path.exists(true_csv_path):
        print("\n⚠ True.csv not found!")
        print("\nThe Kaggle 'Fake and real news dataset' typically has two files:")
        print("  1. Fake.csv - Fake news articles (you have this)")
        print("  2. True.csv - Real news articles (missing)")
        print("\nOptions to get True.csv:")
        print("  1. Download from Kaggle: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset")
        print("  2. Place True.csv in the project root directory")
        print("\nFor now, we'll create a balanced dataset using your sample_news.csv")
        print("and the preprocessed fake news data.")
        
        # Load the sample real news
        sample_path = os.path.join(data_dir, 'sample_news.csv')
        if os.path.exists(sample_path):
            print(f"\n✓ Found sample_news.csv")
            sample_df = pd.read_csv(sample_path)
            real_news = sample_df[sample_df['label'] == 'real']
            print(f"  - Real news samples: {len(real_news)}")
            
            # Load preprocessed fake news
            fake_path = os.path.join(data_dir, 'preprocessed_kaggle_data.csv')
            if os.path.exists(fake_path):
                print(f"\n✓ Loading preprocessed fake news...")
                fake_df = pd.read_csv(fake_path)
                
                # Sample equal amounts
                n_real = len(real_news)
                n_fake = min(len(fake_df), n_real * 10)  # Keep more fake news
                
                fake_sample = fake_df.sample(n=n_fake, random_state=42)
                
                # Combine
                balanced_df = pd.concat([
                    pd.DataFrame({'text': real_news['text'], 'label': 'real'}),
                    pd.DataFrame({'text': fake_sample['text'], 'label': 'fake'})
                ], ignore_index=True)
                
                # Shuffle
                balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
                
                output_path = os.path.join(data_dir, 'balanced_dataset_small.csv')
                balanced_df.to_csv(output_path, index=False)
                
                print(f"\n✓ Created small balanced dataset:")
                print(f"  - Total: {len(balanced_df)} articles")
                print(f"  - Real: {(balanced_df['label']=='real').sum()}")
                print(f"  - Fake: {(balanced_df['label']=='fake').sum()}")
                print(f"  - Saved to: {output_path}")
                print(f"\n⚠ NOTE: This dataset is small. For better accuracy, please download")
                print(f"   True.csv from Kaggle and run this script again.")
        else:
            print(f"\n✗ sample_news.csv not found either!")
            print("Please download True.csv from Kaggle.")
    else:
        # Process True.csv
        print("\n✓ Found True.csv! Processing...")
        
        try:
            true_df = pd.read_csv(true_csv_path, encoding='utf-8')
        except:
            true_df = pd.read_csv(true_csv_path, encoding='latin-1')
        
        print(f"  - Real news articles: {len(true_df)}")
        print(f"  - Columns: {list(true_df.columns)}")
        
        # Process True.csv similar to Fake.csv
        if 'title' in true_df.columns and 'text' in true_df.columns:
            true_df['combined_text'] = true_df['title'].fillna('') + ' ' + true_df['text'].fillna('')
            text_column = 'combined_text'
        elif 'text' in true_df.columns:
            text_column = 'text'
        else:
            text_column = true_df.columns[0]
        
        # Clean
        true_df = true_df.dropna(subset=[text_column])
        true_df = true_df[true_df[text_column].astype(str).str.strip() != '']
        true_df = true_df.drop_duplicates(subset=[text_column])
        
        print(f"  - After cleaning: {len(true_df)} articles")
        
        # Load fake news
        fake_path = os.path.join(data_dir, 'preprocessed_kaggle_data.csv')
        fake_df = pd.read_csv(fake_path)
        
        print(f"\n✓ Loaded fake news: {len(fake_df)} articles")
        
        # Create balanced dataset
        n_samples = min(len(true_df), len(fake_df))
        print(f"\n  - Balancing to {n_samples} samples each...")
        
        true_sample = true_df.sample(n=n_samples, random_state=42)
        fake_sample = fake_df.sample(n=n_samples, random_state=42)
        
        balanced_df = pd.concat([
            pd.DataFrame({'text': true_sample[text_column], 'label': 'real'}),
            pd.DataFrame({'text': fake_sample['text'], 'label': 'fake'})
        ], ignore_index=True)
        
        # Shuffle
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        output_path = os.path.join(data_dir, 'balanced_kaggle_dataset.csv')
        balanced_df.to_csv(output_path, index=False)
        
        print(f"\n✓ Created balanced dataset:")
        print(f"  - Total: {len(balanced_df)} articles")
        print(f"  - Real: {(balanced_df['label']=='real').sum()}")
        print(f"  - Fake: {(balanced_df['label']=='fake').sum()}")
        print(f"  - Saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    create_balanced_dataset()
