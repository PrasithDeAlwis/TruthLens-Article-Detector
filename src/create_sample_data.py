"""
Example dataset creator for Fake News Detection
Creates a sample dataset for testing the model
"""

import pandas as pd
import os

def create_sample_dataset():
    """
    Create a sample fake news dataset for testing
    """
    
    # Sample fake news articles
    fake_articles = [
        "Scientists discover that eating chocolate can make you fly! New research shows amazing results.",
        "BREAKING: Government admits to hiding aliens in secret base. Full story revealed!",
        "Miracle cure found! This simple trick doctors don't want you to know.",
        "Celebrity caught in massive scandal. You won't believe what happened next!",
        "New study proves that water is actually dangerous. Experts shocked!",
        "Local woman discovers secret to eternal youth in her backyard.",
        "URGENT: Your phone is spying on you right now. Here's the proof!",
        "Shocking revelation: Historical event never actually happened.",
        "Scientists baffled by mysterious phenomenon that defies all logic.",
        "Breaking news: Major corporation hiding dangerous truth from public.",
    ]
    
    # Sample real news articles
    real_articles = [
        "New research published in Nature shows promising results in cancer treatment studies.",
        "Stock market shows mixed results as investors await Federal Reserve decision on interest rates.",
        "Local government announces infrastructure improvements scheduled for next quarter.",
        "University study reveals trends in social media usage among teenagers.",
        "Technology company reports quarterly earnings beating analyst expectations.",
        "Climate scientists present new data on global temperature trends at international conference.",
        "City council votes to approve budget for public transportation expansion.",
        "Research team develops new method for detecting early signs of disease.",
        "Economic indicators suggest continued growth in manufacturing sector.",
        "International trade negotiations continue as countries seek agreement on tariffs.",
    ]
    
    # Create dataframe
    data = {
        'text': fake_articles + real_articles,
        'label': ['fake'] * len(fake_articles) + ['real'] * len(real_articles)
    }
    
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def save_sample_dataset(output_dir='../data'):
    """
    Save sample dataset to CSV file
    
    Args:
        output_dir (str): Directory to save dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = create_sample_dataset()
    output_path = os.path.join(output_dir, 'sample_news.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Sample dataset created with {len(df)} articles")
    print(f"Saved to: {output_path}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    return output_path


if __name__ == "__main__":
    save_sample_dataset()
