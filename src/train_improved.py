"""
Improved Training Script for Kaggle Dataset
This script trains the model with optimized parameters for better accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
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


def train_improved_model(data_path, output_dir='../models'):
    """
    Train an improved fake news detection model
    
    Args:
        data_path: Path to the preprocessed dataset
        output_dir: Directory to save the trained model
    """
    print("=" * 80)
    print("IMPROVED FAKE NEWS DETECTION MODEL TRAINING")
    print("=" * 80)
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, 'models')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\n[1/7] Loading preprocessed data...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples")
    print(f"  - Label distribution:")
    print(df['label'].value_counts())
    
    # Check if we have processed_text column or need to process
    if 'processed_text' not in df.columns:
        print("\n[2/7] Preprocessing text...")
        preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
        df['processed_text'] = df['text'].apply(preprocessor.preprocess)
    else:
        print("\n[2/7] Using existing preprocessed text...")
        preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
    
    # Remove empty processed texts
    df = df[df['processed_text'].str.strip() != '']
    df = df.reset_index(drop=True)
    print(f"✓ {len(df)} samples after cleaning")
    
    # Extract features with improved parameters
    print("\n[3/7] Extracting features...")
    print("  - Using TF-IDF with optimized parameters...")
    print("  - Max features: 10000")
    print("  - N-gram range: (1, 3) - includes unigrams, bigrams, and trigrams")
    print("  - Min document frequency: 2")
    
    feature_extractor = FeatureExtractor(
        method='tfidf',
        max_features=10000,  # Increased from 5000
        ngram_range=(1, 3),  # Include trigrams
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95  # Ignore terms that appear in more than 95% of documents
    )
    
    X = feature_extractor.fit_transform(df['processed_text'])
    print(f"✓ Feature matrix shape: {X.shape}")
    
    # Prepare labels
    label_mapping = {'real': 0, 'fake': 1}
    y = df['label'].map(label_mapping).values
    
    # Split data with stratification
    print("\n[4/7] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Training samples: {len(y_train)}")
    print(f"✓ Test samples: {len(y_test)}")
    
    # Train multiple models and create an ensemble
    print("\n[5/7] Training models...")
    print("  - Training Logistic Regression...")
    
    # Logistic Regression with GridSearch
    lr_params = {
        'C': [0.1, 1, 10],
        'max_iter': [1000],
        'solver': ['liblinear', 'saga']
    }
    
    lr = LogisticRegression(random_state=42)
    lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='accuracy', n_jobs=1, verbose=1)
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_
    print(f"    ✓ Best LR params: {lr_grid.best_params_}")
    print(f"    ✓ Best LR CV score: {lr_grid.best_score_:.4f}")
    
    # Multinomial Naive Bayes with GridSearch
    print("\n  - Training Multinomial Naive Bayes...")
    nb_params = {
        'alpha': [0.1, 0.5, 1.0, 2.0]
    }
    
    nb = MultinomialNB()
    nb_grid = GridSearchCV(nb, nb_params, cv=5, scoring='accuracy', n_jobs=1, verbose=1)
    nb_grid.fit(X_train, y_train)
    best_nb = nb_grid.best_estimator_
    print(f"    ✓ Best NB params: {nb_grid.best_params_}")
    print(f"    ✓ Best NB CV score: {nb_grid.best_score_:.4f}")
    
    # Random Forest
    print("\n  - Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1
    )
    rf.fit(X_train, y_train)
    rf_score = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy').mean()
    print(f"    ✓ RF CV score: {rf_score:.4f}")
    
    # Create Voting Classifier (Ensemble)
    print("\n  - Creating ensemble model...")
    ensemble = VotingClassifier(
        estimators=[
            ('lr', best_lr),
            ('nb', best_nb),
            ('rf', rf)
        ],
        voting='soft',
        n_jobs=1
    )
    
    print("  - Training ensemble...")
    ensemble.fit(X_train, y_train)
    print("    ✓ Ensemble training complete!")
    
    # Evaluate all models
    print("\n[6/7] Evaluating models...")
    
    models = {
        'Logistic Regression': best_lr,
        'Naive Bayes': best_nb,
        'Random Forest': rf,
        'Ensemble': ensemble
    }
    
    results = {}
    best_model = None
    best_accuracy = 0
    best_model_name = ''
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        print(f"\n{name}:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
    
    print(f"\n✓ Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    # Detailed evaluation of best model
    print(f"\n[7/7] Detailed evaluation of {best_model_name}...")
    y_pred = best_model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['real', 'fake']))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              Real    Fake")
    print(f"Actual Real   {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"       Fake   {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    # Save the best model
    print("\n[8/8] Saving model...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'best_model_{best_model_name.lower().replace(" ", "_")}_{timestamp}.pkl'
    model_path = os.path.join(output_dir, model_filename)
    
    # Save model components
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
        'training_date': timestamp,
        'dataset_size': len(df),
        'feature_count': X.shape[1],
        'results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
    }
    
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModel Performance:")
    print(f"  - Best Model: {best_model_name}")
    print(f"  - Accuracy: {best_accuracy:.2%}")
    print(f"  - F1 Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"\nModel saved to: {model_path}")
    print(f"\nTo use this model:")
    print(f"  1. Run the web app: python app.py")
    print(f"  2. Or use predict.py for command-line predictions")
    print("=" * 80)
    
    return model_path, best_accuracy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train improved fake news detection model')
    parser.add_argument('--data', type=str, default='../data/preprocessed_kaggle_data.csv',
                        help='Path to preprocessed data')
    parser.add_argument('--output', type=str, default='../models',
                        help='Output directory for model')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if not os.path.isabs(args.data):
        data_path = os.path.join(project_root, args.data.lstrip('../'))
    else:
        data_path = args.data
    
    print(f"Data path: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        print("\nAvailable datasets:")
        data_dir = os.path.join(project_root, 'data')
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    print(f"  - data/{file}")
        else:
            print("  - No data directory found")
        exit(1)
    
    train_improved_model(data_path, args.output)
