"""
Model Training Module for Fake News Detection
Trains multiple ML models and selects the best one
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import TextPreprocessor, load_and_prepare_data
from feature_extraction import FeatureExtractor


class FakeNewsClassifier:
    """
    Fake news classification model trainer
    """
    
    def __init__(self, model_type='logistic_regression'):
        """
        Initialize classifier
        
        Args:
            model_type (str): Type of model to use
        """
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.feature_extractor = None
        self.label_mapping = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on type"""
        models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='liblinear'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='linear',
                random_state=42,
                probability=True
            ),
            'naive_bayes': MultinomialNB(),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
        }
        
        if self.model_type not in models:
            raise ValueError(f"Model type must be one of {list(models.keys())}")
        
        self.model = models[self.model_type]
    
    def prepare_data(self, filepath, text_column='text', label_column='label', 
                     test_size=0.2, random_state=42):
        """
        Load and prepare data for training
        
        Args:
            filepath (str): Path to data file
            text_column (str): Name of text column
            label_column (str): Name of label column
            test_size (float): Test set size
            random_state (int): Random seed
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print("=" * 50)
        print("LOADING AND PREPARING DATA")
        print("=" * 50)
        
        # Load data
        df = load_and_prepare_data(filepath, text_column, label_column)
        
        # Create label mapping
        unique_labels = sorted(df[label_column].unique())
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        print(f"\nLabel mapping: {self.label_mapping}")
        
        # Preprocess text
        print("\n" + "=" * 50)
        print("PREPROCESSING TEXT")
        print("=" * 50)
        self.preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
        df = self.preprocessor.preprocess_dataframe(df, text_column)
        
        # Extract features
        print("\n" + "=" * 50)
        print("EXTRACTING FEATURES")
        print("=" * 50)
        self.feature_extractor = FeatureExtractor(method='tfidf', max_features=5000, ngram_range=(1, 2))
        X = self.feature_extractor.fit_transform(df['processed_text'])
        
        # Map labels to integers
        y = df[label_column].map(self.label_mapping).values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, cv_folds=5):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds (int): Number of cross-validation folds
        """
        print("\n" + "=" * 50)
        print(f"TRAINING {self.model_type.upper()} MODEL")
        print("=" * 50)
        
        # Cross-validation
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model
        print("\nTraining final model...")
        self.model.fit(X_train, y_train)
        print("Training complete!")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        print("\n" + "=" * 50)
        print("EVALUATING MODEL")
        print("=" * 50)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        
        # Get label names
        label_names = {v: k for k, v in self.label_mapping.items()}
        target_names = [label_names[i] for i in sorted(label_names.keys())]
        
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        results = {
            'accuracy': float(accuracy),
            'classification_report': classification_report(y_test, y_pred, target_names=target_names, output_dict=True),
            'confusion_matrix': cm.tolist()
        }
        
        return results
    
    def predict(self, texts):
        """
        Predict labels for new texts
        
        Args:
            texts (list): List of text documents
            
        Returns:
            numpy.ndarray: Predicted labels
        """
        # Preprocess
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # Extract features
        features = self.feature_extractor.transform(processed_texts)
        
        # Predict
        predictions = self.model.predict(features)
        
        # Map back to original labels
        label_names = {v: k for k, v in self.label_mapping.items()}
        return [label_names[pred] for pred in predictions]
    
    def predict_proba(self, texts):
        """
        Predict probabilities for new texts
        
        Args:
            texts (list): List of text documents
            
        Returns:
            numpy.ndarray: Prediction probabilities
        """
        # Preprocess
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # Extract features
        features = self.feature_extractor.transform(processed_texts)
        
        # Predict probabilities
        return self.model.predict_proba(features)
    
    def save(self, model_dir='../models'):
        """
        Save model and associated components
        
        Args:
            model_dir (str): Directory to save model
        """
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.model_type}_{timestamp}"
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        
        # Save model and components
        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_extractor': self.feature_extractor,
            'label_mapping': self.label_mapping,
            'model_type': self.model_type
        }, model_path)
        
        print(f"\nModel saved to {model_path}")
        return model_path
    
    def load(self, model_path):
        """
        Load model and associated components
        
        Args:
            model_path (str): Path to saved model
        """
        data = joblib.load(model_path)
        self.model = data['model']
        self.preprocessor = data['preprocessor']
        self.feature_extractor = data['feature_extractor']
        self.label_mapping = data['label_mapping']
        self.model_type = data['model_type']
        
        print(f"Model loaded from {model_path}")


def train_multiple_models(filepath, text_column='text', label_column='label'):
    """
    Train and compare multiple models
    
    Args:
        filepath (str): Path to data file
        text_column (str): Name of text column
        label_column (str): Name of label column
        
    Returns:
        dict: Results for all models
    """
    model_types = ['logistic_regression', 'random_forest', 'naive_bayes', 'svm']
    results = {}
    
    for model_type in model_types:
        print("\n" + "=" * 70)
        print(f"TRAINING {model_type.upper()} MODEL")
        print("=" * 70)
        
        try:
            classifier = FakeNewsClassifier(model_type=model_type)
            X_train, X_test, y_train, y_test = classifier.prepare_data(
                filepath, text_column, label_column
            )
            classifier.train(X_train, y_train)
            evaluation = classifier.evaluate(X_test, y_test)
            
            results[model_type] = {
                'accuracy': evaluation['accuracy'],
                'model': classifier
            }
            
        except Exception as e:
            print(f"Error training {model_type}: {str(e)}")
            results[model_type] = {'accuracy': 0, 'error': str(e)}
    
    # Find best model
    best_model_type = max(results, key=lambda x: results[x]['accuracy'])
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    for model_type, result in results.items():
        if 'error' not in result:
            print(f"{model_type}: {result['accuracy']:.4f}")
    
    print(f"\nBest model: {best_model_type} with accuracy {results[best_model_type]['accuracy']:.4f}")
    
    return results, best_model_type


if __name__ == "__main__":
    # Example usage
    print("Fake News Detection Model Training")
    print("Please provide a dataset with 'text' and 'label' columns")
