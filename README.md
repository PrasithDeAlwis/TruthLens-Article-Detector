# TruthLens: Fake News Detection System

A machine learning-based system for detecting fake news articles using natural language processing and multiple classification algorithms.

## 🎯 Project Overview

TruthLens is an article-level fake news classification system that uses advanced text preprocessing, TF-IDF feature extraction, and multiple machine learning models to identify potentially fake or misleading news articles.

## ✨ Features

- **Text Preprocessing**: Comprehensive cleaning, tokenization, lemmatization, and stopword removal
- **Feature Extraction**: TF-IDF vectorization with n-gram support
- **Multiple ML Models**: 
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - Naive Bayes
  - Gradient Boosting
- **Model Evaluation**: Detailed metrics, confusion matrices, ROC curves, and comparison visualizations
- **Easy Prediction**: Interactive and batch prediction modes
- **Extensible Architecture**: Modular design for easy customization

## 📁 Project Structure

```
TruthLens-Article-Detector/
├── data/                      # Dataset storage
│   └── sample_news.csv       # Sample dataset
├── models/                    # Trained models
├── src/                       # Source code
│   ├── preprocess.py         # Text preprocessing
│   ├── feature_extraction.py # Feature engineering
│   ├── train.py              # Model training
│   ├── evaluate.py           # Model evaluation
│   ├── predict.py            # Prediction interface
│   ├── create_sample_data.py # Sample data generator
│   └── main.py               # Main training script
├── notebooks/                 # Jupyter notebooks
├── reports/                   # Evaluation reports and plots
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore file
└── README.md                 # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository** (or you're already here!)

2. **Activate the virtual environment**:

```powershell
.\venv\Scripts\Activate
```

3. **Install dependencies**:

```powershell
pip install -r requirements.txt
```

4. **Download NLTK data** (will be done automatically on first run):

The system will automatically download required NLTK datasets (punkt, stopwords, wordnet) when you first run it.

## 📊 Usage

### Training a Model

#### Option 1: Interactive Training (Recommended for beginners)

```powershell
cd src
python main.py
```

Follow the interactive prompts to:
- Choose between single or multiple model training
- Select your preferred algorithm
- View training progress and results

#### Option 2: Direct Training with Custom Dataset

```python
from train import FakeNewsClassifier

# Initialize classifier
classifier = FakeNewsClassifier(model_type='logistic_regression')

# Prepare data (CSV file with 'text' and 'label' columns)
X_train, X_test, y_train, y_test = classifier.prepare_data(
    'path/to/your/data.csv',
    text_column='text',
    label_column='label'
)

# Train model
classifier.train(X_train, y_train)

# Evaluate
results = classifier.evaluate(X_test, y_test)

# Save model
classifier.save()
```

### Making Predictions

#### Interactive Mode

```powershell
cd src
python predict.py --interactive
```

#### Predict Single Article

```powershell
python predict.py --model ../models/model_name.pkl --text "Your article text here"
```

#### Batch Prediction from CSV

```powershell
python predict.py --model ../models/model_name.pkl --file ../data/articles.csv --text-column text --output ../data/predictions.csv
```

#### Python API

```python
from predict import FakeNewsPredictor

# Load model
predictor = FakeNewsPredictor('../models/your_model.pkl')

# Predict single article
prediction, probabilities = predictor.predict_single(
    "Your article text here",
    return_proba=True
)

print(f"Prediction: {prediction}")
print(f"Probabilities: {probabilities}")

# Detailed analysis
analysis = predictor.analyze_article("Your article text here")
```

## 📈 Model Performance

The system trains multiple models and compares their performance. Typical results on balanced datasets:

- **Logistic Regression**: ~85-90% accuracy (fast, interpretable)
- **Random Forest**: ~87-92% accuracy (robust, handles non-linearity)
- **SVM**: ~86-91% accuracy (good for high-dimensional data)
- **Naive Bayes**: ~82-87% accuracy (very fast, baseline)

Performance varies based on dataset quality and size.

## 🔧 Customization

### Using Your Own Dataset

Your dataset should be a CSV file with at least two columns:
- `text`: The article content
- `label`: The classification label (e.g., 'fake', 'real')

Example:
```csv
text,label
"Article text here...",fake
"Another article...",real
```

### Recommended Datasets

For training robust models, consider using:

1. **LIAR Dataset**: 12.8K short statements labeled for truthfulness
2. **ISOT Fake News Dataset**: 44,898 news articles (21,417 real, 23,481 fake)
3. **Kaggle Fake News Dataset**: Various sizes available
4. **FakeNewsNet**: Social context data included

### Adjusting Model Parameters

Edit the model initialization in `train.py`:

```python
# Example: Tune Random Forest
'random_forest': RandomForestClassifier(
    n_estimators=200,      # Increase trees
    max_depth=50,          # Limit depth
    min_samples_split=5,   # Prevent overfitting
    random_state=42
)
```

### Feature Extraction Options

Modify `feature_extraction.py`:

```python
extractor = FeatureExtractor(
    method='tfidf',        # or 'count'
    max_features=10000,    # Increase vocabulary size
    ngram_range=(1, 3)     # Include trigrams
)
```

## 📊 Evaluation Metrics

The system provides:

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions
- **ROC Curve**: For binary classification
- **Feature Importance**: For tree-based models

## 🛠️ Advanced Features

### Cross-Validation

The training script automatically performs 5-fold cross-validation to ensure model generalization.

### Dimensionality Reduction

For very large feature spaces:

```python
from feature_extraction import FeatureExtractor

extractor = FeatureExtractor(max_features=10000)
features = extractor.fit_transform(texts)

# Apply SVD
reduced_features = extractor.apply_dimensionality_reduction(features, n_components=300)
```

### Ensemble Methods

Combine multiple models for better predictions (future enhancement).

## 🐛 Troubleshooting

### Common Issues

**Issue**: NLTK download errors
```
Solution: Run in Python:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

**Issue**: Memory errors with large datasets
```
Solution: Reduce max_features in FeatureExtractor or use dimensionality reduction
```

**Issue**: Low accuracy
```
Solutions:
- Increase dataset size
- Balance classes
- Try different models
- Adjust preprocessing parameters
- Use more features (increase max_features)
```

## 📝 Citation

If you use this project in your research, please cite:

```
TruthLens: Article-Level Fake News Detection System
GitHub: https://github.com/PrasithDeAlwis/TruthLens-Article-Detector
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Deep learning models (LSTM, BERT)
- Additional feature engineering
- Web interface
- API deployment
- More evaluation metrics
- Support for multiple languages

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- scikit-learn for ML algorithms
- NLTK for NLP preprocessing
- The open-source community for fake news datasets

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This system is for educational and research purposes. Always verify news from multiple reliable sources.
