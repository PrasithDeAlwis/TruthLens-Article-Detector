"""
Content-Based Model Training
Trains model to focus on linguistic patterns, not just length
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from datetime import datetime
import os

import sys
import os
sys.path.append('src')
sys.path.append(os.path.dirname(__file__))
from content_features import ContentFeatureExtractor, preprocess_text


def train_content_based_model(data_path='data/balanced_2000_dataset.csv'):
    """
    Train a model that focuses on content, not length
    """
    
    print("=" * 70)
    print("CONTENT-BASED FAKE NEWS DETECTION TRAINING")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Filter out very short texts (they don't have enough content)
    print(f"Original dataset: {len(df)} articles")
    df = df[df['text'].str.len() > 50]  # At least 50 characters
    print(f"After filtering short texts: {len(df)} articles")
    
    X = df['text']
    y = df['label']
    
    print(f"\n📊 Dataset:")
    print(f"   Real: {(y == 'real').sum()}")
    print(f"   Fake: {(y == 'fake').sum()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    print(f"\n✂️  Split:")
    print(f"   Training: {len(X_train)}")
    print(f"   Testing: {len(X_test)}")
    
    # Create feature pipeline
    print("\n🔧 Building content-based feature pipeline...")
    
    feature_pipeline = FeatureUnion([
        # TF-IDF features (word content)
        ('tfidf', TfidfVectorizer(
            max_features=2000,      # Reduce features
            ngram_range=(1, 3),     # Include trigrams for better context
            min_df=3,
            max_df=0.85,
            sublinear_tf=True,
            preprocessor=preprocess_text
        )),
        # Content-based features (linguistic patterns)
        ('content', ContentFeatureExtractor())
    ])
    
    # Build complete pipeline with model
    print("🤖 Training Logistic Regression with content features...")
    
    model = Pipeline([
        ('features', feature_pipeline),
        ('classifier', LogisticRegression(
            max_iter=1000,
            C=1.0,              # Less regularization to capture content patterns
            class_weight='balanced',  # Handle any imbalance
            random_state=42
        ))
    ])
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n📈 Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")
    
    # Cross-validation
    print("\n🔄 Cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
    print(f"   CV Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
    
    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['fake', 'real']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['real', 'fake'])
    print("\n📉 Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Real  Fake")
    print(f"Actual Real   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"       Fake   {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/content_based_model_{timestamp}.pkl'
    
    pipeline_data = {
        'model': model,
        'model_name': 'Content-Based Logistic Regression',
        'features': 'TF-IDF + Linguistic Patterns'
    }
    
    joblib.dump(pipeline_data, model_path)
    
    print(f"\n💾 Model saved: {model_path}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': 'Content-Based Logistic Regression',
        'test_accuracy': float(accuracy),
        'cv_accuracy': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'features': 'TF-IDF + Linguistic Pattern Features',
        'dataset': os.path.basename(data_path)
    }
    
    import json
    with open('models/latest_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    
    return model, model_path


if __name__ == "__main__":
    train_content_based_model()
