"""
Download True.csv from a mirror or instruct user to download
"""

import os
import sys

def check_and_download_true_csv():
    """
    Check for True.csv and provide instructions to download it
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    true_csv_path = os.path.join(project_root, 'True.csv')
    
    print("=" * 80)
    print("KAGGLE FAKE NEWS DATASET - TRUE.CSV CHECK")
    print("=" * 80)
    
    if os.path.exists(true_csv_path):
        print("\n✓ True.csv found!")
        print(f"  Location: {true_csv_path}")
        
        import pandas as pd
        try:
            df = pd.read_csv(true_csv_path)
            print(f"  Size: {len(df)} articles")
            return True
        except Exception as e:
            print(f"  Error reading file: {e}")
            return False
    else:
        print("\n✗ True.csv NOT found!")
        print("\nThe complete Kaggle 'Fake and Real News Dataset' has TWO files:")
        print("  1. Fake.csv - Fake news articles (✓ You have this)")
        print("  2. True.csv - Real news articles (✗ Missing)")
        print("\nTO GET TRUE.CSV:")
        print("\nOption 1: Download from Kaggle (Recommended)")
        print("-" * 80)
        print("1. Go to: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print("2. Click 'Download' button (you may need to sign in)")
        print("3. Extract the archive.zip file")
        print("4. Copy 'True.csv' to your project root directory:")
        print(f"   {project_root}")
        print("\nOption 2: Use alternative dataset")
        print("-" * 80)
        print("Use a different balanced dataset or manually create one.")
        
        print("\n" + "=" * 80)
        print("ONCE YOU HAVE TRUE.CSV:")
        print("=" * 80)
        print("\n1. Place True.csv in:")
        print(f"   {project_root}\\True.csv")
        print("\n2. Run preprocessing:")
        print("   python src/preprocess_kaggle_data.py")
        print("\n3. Train the improved model:")
        print("   python src/train_improved.py --data data/preprocessed_kaggle_data.csv")
        print("\n" + "=" * 80)
        
        return False


if __name__ == "__main__":
    check_and_download_true_csv()
