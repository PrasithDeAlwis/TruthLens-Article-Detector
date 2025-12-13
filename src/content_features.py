"""
Enhanced Feature Extraction for Content-Based Fake News Detection
Focuses on linguistic patterns, not just length
"""

import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


class ContentFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract content-based features from text
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Extract features that focus on content patterns
        """
        features = []
        
        for text in X:
            text_features = {}
            
            # Emotional/sensational language indicators
            text_features['exclamation_marks'] = text.count('!')
            text_features['question_marks'] = text.count('?')
            text_features['all_caps_words'] = len([w for w in text.split() if w.isupper() and len(w) > 1])
            
            # Clickbait indicators
            clickbait_words = ['shocking', 'breaking', 'unbelievable', 'you wont believe', 
                             'doctors hate', 'one weird trick', 'what happened next']
            text_lower = text.lower()
            text_features['clickbait_score'] = sum(1 for word in clickbait_words if word in text_lower)
            
            # Bias indicators
            bias_words = ['always', 'never', 'everyone', 'nobody', 'absolutely', 
                         'totally', 'completely', 'obviously', 'clearly']
            text_features['bias_words'] = sum(1 for word in bias_words if word in text_lower)
            
            # Emotional language
            emotional_words = ['amazing', 'terrible', 'horrible', 'disgusting', 'outrageous',
                             'shocking', 'unbelievable', 'incredible', 'devastating']
            text_features['emotional_words'] = sum(1 for word in emotional_words if word in text_lower)
            
            # Quotation usage (fake news often uses quotes out of context)
            text_features['quotes_count'] = text.count('"') + text.count("'")
            
            # Uppercase ratio (shouting)
            if len(text) > 0:
                text_features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
            else:
                text_features['uppercase_ratio'] = 0
            
            # Number of numbers (fake news often uses specific numbers for credibility)
            text_features['numbers_count'] = len(re.findall(r'\d+', text))
            
            # Average word length (complex vs simple language)
            words = text.split()
            if words:
                text_features['avg_word_length'] = np.mean([len(w) for w in words])
            else:
                text_features['avg_word_length'] = 0
            
            # Normalize by text length to avoid length bias
            word_count = len(words) if words else 1
            text_features['exclamation_density'] = text_features['exclamation_marks'] / word_count
            text_features['question_density'] = text_features['question_marks'] / word_count
            text_features['caps_density'] = text_features['all_caps_words'] / word_count
            text_features['clickbait_density'] = text_features['clickbait_score'] / word_count
            text_features['bias_density'] = text_features['bias_words'] / word_count
            text_features['emotional_density'] = text_features['emotional_words'] / word_count
            
            features.append(text_features)
        
        # Convert to numpy array
        import pandas as pd
        df = pd.DataFrame(features)
        return df.values


def preprocess_text(text):
    """
    Clean text while preserving important content signals
    """
    # Keep emotional indicators (!, ?, CAPS) for feature extraction
    # But normalize for TF-IDF
    
    # Convert to lowercase for TF-IDF (but we extracted caps info first)
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text
