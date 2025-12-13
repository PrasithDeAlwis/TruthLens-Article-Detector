"""
Prediction Script for Fake News Detection
Make predictions on new articles using trained model
"""

import sys
import os
import joblib
import pandas as pd
import re
import string

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class FakeNewsPredictor:
    """
    Make predictions on new articles
    """
    
    def __init__(self, model_path):
        """
        Initialize predictor with trained model
        
        Args:
            model_path (str): Path to saved model
        """
        # Load the model
        pipeline = joblib.load(model_path)
        
        # Check model type
        if isinstance(pipeline, dict):
            # Check if it's a content-based model (has 'model' key as pipeline)
            if hasattr(pipeline.get('model'), 'predict'):
                self.model = pipeline['model']
                self.model_name = pipeline.get('model_name', 'Unknown')
                self.is_content_model = True
                self.is_notebook_model = False
            else:
                # Notebook-trained model (dict with vectorizer and model)
                self.vectorizer = pipeline['vectorizer']
                self.model = pipeline['model']
                self.model_name = pipeline.get('model_name', 'Unknown')
                self.is_notebook_model = True
                self.is_content_model = False
        else:
            # Old format compatibility
            from train import FakeNewsClassifier
            self.classifier = FakeNewsClassifier()
            self.classifier.load(model_path)
            self.is_notebook_model = False
            self.is_content_model = False
        
        print(f"Model loaded successfully from {model_path}")
        if self.is_notebook_model or self.is_content_model:
            print(f"Model type: {self.model_name}")
    
    def clean_text(self, text):
        """
        Clean and preprocess text (same as training)
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    
    def predict_single(self, text, return_proba=False):
        """
        Predict label for a single article
        
        Args:
            text (str): Article text
            return_proba (bool): Return probabilities
            
        Returns:
            str or tuple: Predicted label (and probabilities if requested)
        """
        if self.is_content_model:
            # Content-based model (pipeline handles everything)
            prediction = self.model.predict([text])[0]
            
            if return_proba:
                proba = self.model.predict_proba([text])[0]
                classes = self.model.classes_
                proba_dict = {label: float(prob) for label, prob in zip(classes, proba)}
                return prediction, proba_dict
            
            return prediction
            
        elif self.is_notebook_model:
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            # Transform using vectorizer
            X = self.vectorizer.transform([cleaned_text])
            
            # Predict
            prediction = self.model.predict(X)[0]
            
            if return_proba:
                proba = self.model.predict_proba(X)[0]
                # Get class labels
                classes = self.model.classes_
                proba_dict = {label: float(prob) for label, prob in zip(classes, proba)}
                return prediction, proba_dict
            
            return prediction
        else:
            # Use old classifier method
            prediction = self.classifier.predict([text])[0]
            
            if return_proba:
                proba = self.classifier.predict_proba([text])[0]
                label_names = {v: k for k, v in self.classifier.label_mapping.items()}
                proba_dict = {label_names[i]: float(prob) for i, prob in enumerate(proba)}
                return prediction, proba_dict
            
            return prediction
    
    def predict_batch(self, texts, return_proba=False):
        """
        Predict labels for multiple articles
        
        Args:
            texts (list): List of article texts
            return_proba (bool): Return probabilities
            
        Returns:
            list or tuple: Predicted labels (and probabilities if requested)
        """
        if self.is_content_model:
            # Content-based model (pipeline handles everything)
            predictions = self.model.predict(texts)
            
            if return_proba:
                probas = self.model.predict_proba(texts)
                classes = self.model.classes_
                
                proba_dicts = []
                for proba in probas:
                    proba_dict = {label: float(prob) for label, prob in zip(classes, proba)}
                    proba_dicts.append(proba_dict)
                
                return predictions, proba_dicts
            
            return predictions
            
        elif self.is_notebook_model:
            # Clean all texts
            cleaned_texts = [self.clean_text(text) for text in texts]
            
            # Transform using vectorizer
            X = self.vectorizer.transform(cleaned_texts)
            
            # Predict
            predictions = self.model.predict(X)
            
            if return_proba:
                probas = self.model.predict_proba(X)
                classes = self.model.classes_
                
                proba_dicts = []
                for proba in probas:
                    proba_dict = {label: float(prob) for label, prob in zip(classes, proba)}
                    proba_dicts.append(proba_dict)
                
                return predictions, proba_dicts
            
            return predictions
        else:
            # Use old classifier method
            predictions = self.classifier.predict(texts)
            
            if return_proba:
                probas = self.classifier.predict_proba(texts)
                label_names = {v: k for k, v in self.classifier.label_mapping.items()}
                
                proba_dicts = []
                for proba in probas:
                    proba_dict = {label_names[i]: float(prob) for i, prob in enumerate(proba)}
                    proba_dicts.append(proba_dict)
                
                return predictions, proba_dicts
            
            return predictions
    
    def predict_from_file(self, filepath, text_column='text', output_path=None):
        """
        Predict labels for articles in a CSV file
        
        Args:
            filepath (str): Path to input CSV file
            text_column (str): Name of text column
            output_path (str): Path to save results (optional)
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in file")
        
        print(f"Making predictions for {len(df)} articles...")
        predictions, probas = self.predict_batch(df[text_column].tolist(), return_proba=True)
        
        df['predicted_label'] = predictions
        
        # Add probability columns
        if self.is_notebook_model:
            # Get class labels from model
            classes = self.model.classes_
            for label_name in classes:
                df[f'prob_{label_name}'] = [proba[label_name] for proba in probas]
        else:
            # Use old classifier method
            label_names = {v: k for k, v in self.classifier.label_mapping.items()}
            for label_name in label_names.values():
                df[f'prob_{label_name}'] = [proba[label_name] for proba in probas]
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        return df
    
    def analyze_article(self, text):
        """
        Provide detailed analysis of an article
        
        Args:
            text (str): Article text
            
        Returns:
            dict: Detailed analysis
        """
        prediction, proba = self.predict_single(text, return_proba=True)
        
        analysis = {
            'prediction': prediction,
            'confidence': max(proba.values()),
            'probabilities': proba,
            'text_length': len(text),
            'word_count': len(text.split())
        }
        
        return analysis


def interactive_prediction():
    """
    Interactive mode for making predictions
    """
    print("=" * 60)
    print("FAKE NEWS DETECTOR - INTERACTIVE MODE")
    print("=" * 60)
    
    # Get model path
    model_dir = '../models'
    if not os.path.exists(model_dir):
        print(f"Error: Models directory '{model_dir}' not found")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not model_files:
        print(f"Error: No model files found in '{model_dir}'")
        return
    
    print("\nAvailable models:")
    for i, model_file in enumerate(model_files, 1):
        print(f"{i}. {model_file}")
    
    choice = input("\nSelect a model (enter number): ")
    try:
        model_path = os.path.join(model_dir, model_files[int(choice) - 1])
    except (ValueError, IndexError):
        print("Invalid choice")
        return
    
    # Load predictor
    predictor = FakeNewsPredictor(model_path)
    
    print("\n" + "=" * 60)
    print("Enter articles to analyze (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        print("\n" + "-" * 60)
        text = input("\nEnter article text: ")
        
        if text.lower() == 'quit':
            break
        
        if not text.strip():
            print("Please enter some text")
            continue
        
        # Analyze article
        analysis = predictor.analyze_article(text)
        
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Prediction: {analysis['prediction']}")
        print(f"Confidence: {analysis['confidence']:.2%}")
        print(f"\nProbabilities:")
        for label, prob in analysis['probabilities'].items():
            print(f"  {label}: {prob:.2%}")
        print(f"\nText Statistics:")
        print(f"  Length: {analysis['text_length']} characters")
        print(f"  Word Count: {analysis['word_count']} words")


def main():
    """
    Main function for command-line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Fake News Detection Prediction')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--text', type=str, help='Single article text to predict')
    parser.add_argument('--file', type=str, help='CSV file with articles')
    parser.add_argument('--text-column', type=str, default='text', help='Name of text column in CSV')
    parser.add_argument('--output', type=str, help='Output CSV file for predictions')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_prediction()
        return
    
    # Load predictor
    predictor = FakeNewsPredictor(args.model)
    
    if args.text:
        # Predict single text
        analysis = predictor.analyze_article(args.text)
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Prediction: {analysis['prediction']}")
        print(f"Confidence: {analysis['confidence']:.2%}")
        print(f"\nProbabilities:")
        for label, prob in analysis['probabilities'].items():
            print(f"  {label}: {prob:.2%}")
    
    elif args.file:
        # Predict from file
        df = predictor.predict_from_file(args.file, args.text_column, args.output)
        print("\nPredictions:")
        print(df[['predicted_label']].value_counts())
    
    else:
        print("Please provide either --text or --file argument")


if __name__ == "__main__":
    # If no arguments provided, run interactive mode
    if len(sys.argv) == 1:
        interactive_prediction()
    else:
        main()
