"""
Data Preprocessing Module for Fake News Detection
Handles text cleaning, tokenization, and preprocessing
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd

# Download required NLTK data
def download_nltk_data():
    """Download necessary NLTK datasets"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')


class TextPreprocessor:
    """
    A class to handle text preprocessing for fake news detection
    """
    
    def __init__(self, use_lemmatization=True, remove_stopwords=True):
        """
        Initialize the preprocessor
        
        Args:
            use_lemmatization (bool): Use lemmatization instead of stemming
            remove_stopwords (bool): Remove stopwords from text
        """
        download_nltk_data()
        
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        
        if use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.stemmer = PorterStemmer()
        
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Clean and normalize text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords_func(self, tokens):
        """
        Remove stopwords from token list
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Filtered tokens
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """
        Lemmatize tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem_tokens(self, tokens):
        """
        Stem tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stopwords_func(tokens)
        
        # Lemmatize or stem
        if self.use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        else:
            tokens = self.stem_tokens(tokens)
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column):
        """
        Preprocess text column in a dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of the text column
            
        Returns:
            pd.DataFrame: Dataframe with preprocessed text
        """
        df = df.copy()
        df['processed_text'] = df[text_column].apply(self.preprocess)
        return df


def load_and_prepare_data(filepath, text_column='text', label_column='label'):
    """
    Load and prepare data for training
    
    Args:
        filepath (str): Path to the data file
        text_column (str): Name of the text column
        label_column (str): Name of the label column
        
    Returns:
        pd.DataFrame: Prepared dataframe
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Check for missing values
    print(f"Missing values before cleaning:")
    print(df.isnull().sum())
    
    # Drop rows with missing text or labels
    df = df.dropna(subset=[text_column, label_column])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Label distribution:\n{df[label_column].value_counts()}")
    
    return df


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
    
    sample_text = "Breaking News: This is a FAKE article with URLs http://example.com and emails test@test.com!!!"
    processed = preprocessor.preprocess(sample_text)
    
    print("Original text:", sample_text)
    print("Processed text:", processed)
