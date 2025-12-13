"""
Expand dataset to 1000 rows with super clean preprocessing
"""

import pandas as pd
import os
from preprocess import TextPreprocessor
from nltk.corpus import stopwords

def expand_to_1000_rows():
    """Expand dataset to 1000 rows"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    print("=" * 80)
    print("EXPANDING DATASET TO 1000 ROWS")
    print("=" * 80)
    
    # Load large fake news dataset
    print(f"\n[1/5] Loading preprocessed Kaggle data...")
    kaggle_path = os.path.join(data_dir, 'preprocessed_kaggle_data.csv')
    kaggle_df = pd.read_csv(kaggle_path)
    print(f"✓ Available fake news: {len(kaggle_df)}")
    
    # Load existing real news
    balanced_path = os.path.join(data_dir, 'balanced_training_data.csv')
    balanced_df = pd.read_csv(balanced_path)
    real_df = balanced_df[balanced_df['label'] == 'real']
    print(f"✓ Available real news: {len(real_df)}")
    
    # Sample 500 from each
    print(f"\n[2/5] Sampling articles...")
    n_samples = 500
    
    # Get fake news samples
    fake_sample = kaggle_df.sample(n=n_samples, random_state=42)
    fake_sample['label'] = 'fake'
    
    # For real news, duplicate and vary if needed
    if len(real_df) < n_samples:
        print(f"  ⚠ Only {len(real_df)} real news available")
        print(f"  - Using all {len(real_df)} real news samples")
        real_sample = real_df
        # Adjust fake news to match
        fake_sample = kaggle_df.sample(n=len(real_df), random_state=42)
        fake_sample['label'] = 'fake'
    else:
        real_sample = real_df.sample(n=n_samples, random_state=42)
    
    # Combine
    df = pd.concat([real_sample, fake_sample], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✓ Total articles to process: {len(df)}")
    print(f"  - Real: {(df['label']=='real').sum()}")
    print(f"  - Fake: {(df['label']=='fake').sum()}")
    
    # Super clean preprocessing
    print(f"\n[3/5] SUPER CLEANING all articles...")
    print("  - Removing ALL common words")
    print("  - Keeping only meaningful content")
    
    # Extended stopwords
    stop_words = set(stopwords.words('english'))
    extra_stopwords = {
        'of', 'to', 'like', 'we', 'us', 'all', 'also', 'would', 'could', 
        'should', 'may', 'might', 'must', 'will', 'shall', 'can',
        'get', 'got', 'getting', 'go', 'going', 'went', 'gone',
        'say', 'said', 'saying', 'says', 'tell', 'told', 'telling',
        'make', 'made', 'making', 'makes', 'take', 'took', 'taking',
        'know', 'knew', 'known', 'knowing', 'think', 'thought', 'thinking',
        'see', 'saw', 'seen', 'seeing', 'come', 'came', 'coming',
        'want', 'wanted', 'wanting', 'use', 'used', 'using',
        'find', 'found', 'finding', 'give', 'gave', 'given', 'giving',
        'work', 'worked', 'working', 'call', 'called', 'calling',
        'try', 'tried', 'trying', 'ask', 'asked', 'asking',
        'need', 'needed', 'needing', 'feel', 'felt', 'feeling',
        'become', 'became', 'becoming', 'leave', 'left', 'leaving',
        'put', 'putting', 'mean', 'meant', 'meaning', 'keep', 'kept', 'keeping',
        'let', 'letting', 'begin', 'began', 'begun', 'beginning',
        'seem', 'seemed', 'seeming', 'help', 'helped', 'helping',
        'talk', 'talked', 'talking', 'turn', 'turned', 'turning',
        'start', 'started', 'starting', 'show', 'showed', 'shown', 'showing',
        'hear', 'heard', 'hearing', 'play', 'played', 'playing',
        'run', 'ran', 'running', 'move', 'moved', 'moving',
        'live', 'lived', 'living', 'believe', 'believed', 'believing',
        'bring', 'brought', 'bringing', 'happen', 'happened', 'happening',
        'write', 'wrote', 'written', 'writing', 'sit', 'sat', 'sitting',
        'stand', 'stood', 'standing', 'lose', 'lost', 'losing',
        'pay', 'paid', 'paying', 'meet', 'met', 'meeting',
        'include', 'included', 'including', 'continue', 'continued', 'continuing',
        'set', 'setting', 'learn', 'learned', 'learning',
        'change', 'changed', 'changing', 'lead', 'led', 'leading',
        'understand', 'understood', 'understanding', 'watch', 'watched', 'watching',
        'follow', 'followed', 'following', 'stop', 'stopped', 'stopping',
        'create', 'created', 'creating', 'speak', 'spoke', 'spoken', 'speaking',
        'read', 'reading', 'allow', 'allowed', 'allowing',
        'add', 'added', 'adding', 'spend', 'spent', 'spending',
        'grow', 'grew', 'grown', 'growing', 'open', 'opened', 'opening',
        'walk', 'walked', 'walking', 'win', 'won', 'winning',
        'offer', 'offered', 'offering', 'remember', 'remembered', 'remembering',
        'love', 'loved', 'loving', 'consider', 'considered', 'considering',
        'appear', 'appeared', 'appearing', 'buy', 'bought', 'buying',
        'wait', 'waited', 'waiting', 'serve', 'served', 'serving',
        'die', 'died', 'dying', 'send', 'sent', 'sending',
        'expect', 'expected', 'expecting', 'build', 'built', 'building',
        'stay', 'stayed', 'staying', 'fall', 'fell', 'fallen', 'falling',
        'cut', 'cutting', 'reach', 'reached', 'reaching',
        'kill', 'killed', 'killing', 'remain', 'remained', 'remaining',
        'suggest', 'suggested', 'suggesting', 'raise', 'raised', 'raising',
        'pass', 'passed', 'passing', 'sell', 'sold', 'selling',
        'require', 'required', 'requiring', 'report', 'reported', 'reporting',
        'decide', 'decided', 'deciding', 'pull', 'pulled', 'pulling',
        'one', 'two', 'three', 'first', 'second', 'third', 'last', 'next',
        'much', 'many', 'more', 'most', 'some', 'any', 'every', 'each',
        'own', 'well', 'even', 'still', 'just', 'back', 'way', 'now',
        'around', 'today', 'however', 'really', 'thing', 'things',
        'something', 'someone', 'anything', 'anyone', 'everything', 'everyone',
        'nothing', 'nobody', 'somewhere', 'anywhere', 'everywhere', 'nowhere',
        'though', 'although', 'since', 'because', 'while', 'whether',
        'either', 'neither', 'both', 'another', 'other', 'others',
        'quite', 'rather', 'almost', 'nearly', 'hardly', 'barely',
        'yet', 'already', 'always', 'never', 'sometimes', 'often',
        'usually', 'generally', 'actually', 'probably', 'perhaps', 'maybe'
    }
    stop_words.update(extra_stopwords)
    
    preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
    
    cleaned_texts = []
    for i, text in enumerate(df['text']):
        if (i + 1) % 100 == 0:
            print(f"  - Progress: {i+1}/{len(df)}")
        
        # Preprocess
        cleaned = preprocessor.preprocess(text)
        
        # Extra cleaning
        words = cleaned.split()
        words = [w for w in words if w not in stop_words and len(w) > 2]
        cleaned_texts.append(' '.join(words))
    
    df['cleaned_text'] = cleaned_texts
    
    # Remove empty texts
    print(f"\n[4/5] Removing empty/short texts...")
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    df = df[df['word_count'] >= 5]
    df = df.drop('word_count', axis=1)
    
    print(f"✓ Final count: {len(df)} articles")
    print(f"  - Real: {(df['label']=='real').sum()}")
    print(f"  - Fake: {(df['label']=='fake').sum()}")
    
    # Save
    print(f"\n[5/5] Saving expanded dataset...")
    final_df = df[['text', 'cleaned_text', 'label']]
    
    output_path = os.path.join(data_dir, 'expanded_1000_dataset.csv')
    final_df.to_csv(output_path, index=False)
    
    print(f"✓ Saved to: {output_path}")
    
    # Statistics
    print(f"\n" + "=" * 80)
    print("FINAL DATASET STATISTICS")
    print("=" * 80)
    print(f"  Total articles: {len(final_df)}")
    print(f"  Real news: {(final_df['label']=='real').sum()}")
    print(f"  Fake news: {(final_df['label']=='fake').sum()}")
    print(f"  Balance: {(final_df['label']=='real').sum() / len(final_df) * 100:.1f}% real")
    print(f"  Avg words (original): {final_df['text'].str.split().str.len().mean():.0f}")
    print(f"  Avg words (cleaned): {final_df['cleaned_text'].str.split().str.len().mean():.0f}")
    
    print(f"\n" + "=" * 80)
    print("✅ DATASET EXPANDED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nTrain with expanded dataset:")
    print(f"  python src/quick_train_improved.py --data data/expanded_1000_dataset.csv")
    print("=" * 80)
    
    return output_path

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    expand_to_1000_rows()
