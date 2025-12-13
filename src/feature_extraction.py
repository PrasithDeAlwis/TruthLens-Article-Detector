"""
Feature Extraction Module for Fake News Detection
Implements TF-IDF vectorization and other feature extraction methods
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
import joblib
import os


class FeatureExtractor:
    """
    Feature extraction for text data
    """
    
    def __init__(self, method='tfidf', max_features=5000, ngram_range=(1, 2), 
                 min_df=2, max_df=0.95):
        """
        Initialize feature extractor
        
        Args:
            method (str): 'tfidf' or 'count'
            max_features (int): Maximum number of features
            ngram_range (tuple): Range of n-grams to extract
            min_df (int/float): Minimum document frequency
            max_df (float): Maximum document frequency
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.svd = None
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=True
            )
        elif method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df
            )
        else:
            raise ValueError("Method must be 'tfidf' or 'count'")
    
    def fit_transform(self, texts):
        """
        Fit vectorizer and transform texts
        
        Args:
            texts (list): List of text documents
            
        Returns:
            numpy.ndarray: Feature matrix
        """
        print(f"Extracting features using {self.method} vectorization...")
        features = self.vectorizer.fit_transform(texts)
        print(f"Feature matrix shape: {features.shape}")
        return features
    
    def transform(self, texts):
        """
        Transform texts using fitted vectorizer
        
        Args:
            texts (list): List of text documents
            
        Returns:
            numpy.ndarray: Feature matrix
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be fitted first")
        return self.vectorizer.transform(texts)
    
    def apply_dimensionality_reduction(self, features, n_components=300):
        """
        Apply SVD for dimensionality reduction
        
        Args:
            features: Feature matrix
            n_components (int): Number of components to keep
            
        Returns:
            numpy.ndarray: Reduced feature matrix
        """
        print(f"Applying dimensionality reduction to {n_components} components...")
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_features = self.svd.fit_transform(features)
        print(f"Explained variance ratio: {self.svd.explained_variance_ratio_.sum():.4f}")
        return reduced_features
    
    def get_feature_names(self):
        """
        Get feature names from vectorizer
        
        Returns:
            list: Feature names
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be fitted first")
        return self.vectorizer.get_feature_names_out()
    
    def get_top_features(self, n=20):
        """
        Get top features by IDF value (for TF-IDF)
        
        Args:
            n (int): Number of top features to return
            
        Returns:
            list: Top feature names
        """
        if self.vectorizer is None or self.method != 'tfidf':
            raise ValueError("Vectorizer must be fitted and method must be 'tfidf'")
        
        feature_names = self.get_feature_names()
        idf_values = self.vectorizer.idf_
        
        top_indices = np.argsort(idf_values)[-n:]
        return [feature_names[i] for i in top_indices]
    
    def save(self, filepath):
        """
        Save vectorizer to file
        
        Args:
            filepath (str): Path to save vectorizer
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'vectorizer': self.vectorizer,
            'svd': self.svd,
            'method': self.method,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range
        }, filepath)
        print(f"Feature extractor saved to {filepath}")
    
    def load(self, filepath):
        """
        Load vectorizer from file
        
        Args:
            filepath (str): Path to load vectorizer from
        """
        data = joblib.load(filepath)
        self.vectorizer = data['vectorizer']
        self.svd = data.get('svd')
        self.method = data['method']
        self.max_features = data['max_features']
        self.ngram_range = data['ngram_range']
        print(f"Feature extractor loaded from {filepath}")


def extract_text_statistics(df, text_column='processed_text'):
    """
    Extract statistical features from text
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of text column
        
    Returns:
        pd.DataFrame: Dataframe with statistical features
    """
    df = df.copy()
    
    # Text length features
    df['text_length'] = df[text_column].apply(len)
    df['word_count'] = df[text_column].apply(lambda x: len(x.split()))
    df['avg_word_length'] = df[text_column].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
    )
    
    # Punctuation and special characters
    df['exclamation_count'] = df[text_column].apply(lambda x: x.count('!'))
    df['question_count'] = df[text_column].apply(lambda x: x.count('?'))
    df['uppercase_ratio'] = df[text_column].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    
    return df


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "this is a fake news article",
        "this is a real news article",
        "breaking news fake information",
    ]
    
    extractor = FeatureExtractor(method='tfidf', max_features=100)
    features = extractor.fit_transform(sample_texts)
    
    print("Feature matrix:")
    print(features.toarray())
    print("\nFeature names:", extractor.get_feature_names()[:10])
