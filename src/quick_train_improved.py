"""
Quick training with improved hyperparameters for better accuracy
Optimized for faster training while maintaining high accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib
import os
import json
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import TextPreprocessor
from feature_extraction import FeatureExtractor


def quick_train_improved(data_path='../data/balanced_training_data.csv', sample_size=None):
    """
    Quick training with improved parameters
    """
    print("=" * 80)
    print("QUICK TRAINING - IMPROVED MODEL")
    print("=" * 80)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Resolve path
    if not os.path.isabs(data_path):
        data_path = os.path.join(project_root, data_path.lstrip('../'))
    
    # Load data
    print("\n[1/5] Loading data...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples")
    print(f"  Label distribution:\n{df['label'].value_counts()}")
    
    # Sample for quick training if needed
    if sample_size and len(df) > sample_size:
        print(f"\n  Sampling {sample_size} samples for quick training...")
        # Try to balance the sample
        fake_df = df[df['label'] == 'fake']
        real_df = df[df['label'] == 'real']
        
        n_each = sample_size // 2
        n_fake = min(len(fake_df), n_each)
        n_real = min(len(real_df), sample_size - n_fake)
        
        df = pd.concat([
            fake_df.sample(n=n_fake, random_state=42),
            real_df.sample(n=n_real, random_state=42)
        ])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  ✓ Sampled dataset: {len(df)} samples")
        print(f"    - Real: {(df['label']=='real').sum()}")
        print(f"    - Fake: {(df['label']=='fake').sum()}")
    
    # Preprocess if needed
    print("\n[2/5] Preprocessing text...")
    preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
    
    if 'processed_text' not in df.columns:
        print("  Processing text...")
        df['processed_text'] = df['text'].apply(preprocessor.preprocess)
    else:
        print("  Using existing processed text...")
    
    # Remove empty texts
    df = df[df['processed_text'].str.strip() != '']
    print(f"✓ {len(df)} samples after cleaning")
    
    # Extract features
    print("\n[3/5] Extracting features...")
    feature_extractor = FeatureExtractor(
        method='tfidf',
        max_features=5000,  # Reduced for speed
        ngram_range=(1, 2),  # Bigrams only
        min_df=2,
        max_df=0.95
    )
    
    X = feature_extractor.fit_transform(df['processed_text'])
    print(f"✓ Feature matrix: {X.shape}")
    
    # Prepare labels
    label_mapping = {'real': 0, 'fake': 1}
    y = df['label'].map(label_mapping).values
    
    # Split data
    print("\n[4/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Train models
    print("\n[5/5] Training models...")
    
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver='liblinear',
            random_state=42
        ),
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=1  # Changed to 1 to avoid joblib issue
        )
    }
    
    results = {}
    best_model = None
    best_accuracy = 0
    best_model_name = ''
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"    ✓ Accuracy: {accuracy:.4f}")
        print(f"    ✓ F1 Score: {f1:.4f}")
        
        results[name] = {'accuracy': accuracy, 'f1_score': f1}
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
    
    # Detailed evaluation
    print(f"\n" + "=" * 80)
    print(f"BEST MODEL: {best_model_name}")
    print("=" * 80)
    
    y_pred = best_model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['real', 'fake']))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              Real    Fake")
    print(f"Actual Real   {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"       Fake   {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    # Save model
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    
    model_dir = os.path.join(project_root, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'quick_trained_model_{timestamp}.pkl'
    model_path = os.path.join(model_dir, model_filename)
    
    model_data = {
        'model': best_model,
        'preprocessor': preprocessor,
        'feature_extractor': feature_extractor,
        'label_mapping': label_mapping,
        'model_name': best_model_name,
        'accuracy': best_accuracy,
        'results': results
    }
    
    joblib.dump(model_data, model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'accuracy': float(best_accuracy),
        'f1_score': float(results[best_model_name]['f1_score']),
        'training_date': timestamp,
        'dataset_size': len(df),
        'feature_count': X.shape[1],
        'all_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
    }
    
    metadata_path = os.path.join(model_dir, 'latest_model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nFinal Results:")
    print(f"  - Best Model: {best_model_name}")
    print(f"  - Accuracy: {best_accuracy:.2%}")
    print(f"  - F1 Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"\nModel saved to:")
    print(f"  {model_path}")
    print(f"\nTo use the model:")
    print(f"  1. Run: python app.py")
    print(f"  2. Open: http://localhost:5000")
    print("=" * 80)
    
    return model_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/balanced_training_data.csv')
    parser.add_argument('--sample', type=int, default=None, help='Sample size for quick training')
    
    args = parser.parse_args()
    
    quick_train_improved(args.data, args.sample)
