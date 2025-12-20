# TruthLens Project Structure

## 📂 Directory Overview

```
TruthLens-Article-Detector/
├── app.py                              # Flask web application (main entry point)
├── requirements.txt                    # Python package dependencies
├── README.md                           # Main documentation
├── WEB_APP_README.md                   # Web application guide
├── BALANCED_MODEL_RESULTS.md           # Model performance results
├── DATASET_PROCESSING_README.md        # Dataset preparation guide
├── Model_Training.ipynb                # Jupyter notebook for training
│
├── data/                               # Training and test datasets
│   └── balanced_2000_dataset.csv      # Balanced dataset (used by best model)
│
├── models/                             # Trained model files
│   ├── content_based_model_*.pkl      # Best model (99.7% accuracy) ⭐
│   └── latest_model_metadata.json     # Model performance metrics
│
├── src/                                # Core application source code
│   ├── predict.py                     # Prediction module (multi-format support)
│   ├── train.py                       # Model training pipeline
│   ├── preprocess.py                  # Text preprocessing & cleaning
│   ├── feature_extraction.py          # TF-IDF feature engineering
│   ├── content_features.py            # Advanced linguistic features
│   ├── evaluate.py                    # Model evaluation & metrics
│   └── main.py                        # Training orchestration script
│
├── static/                             # Web application assets
│   ├── css/
│   │   └── style.css                  # Application styling
│   └── js/
│       ├── main.js                    # Main JavaScript
│       └── predict.js                 # Prediction interface logic
│
└── templates/                          # HTML templates
    ├── base.html                      # Base template
    ├── index.html                     # Home page & prediction interface
    ├── about.html                     # About page
    ├── 404.html                       # Error page
    └── 500.html                       # Server error page
```

## 🎯 Core Components

### **app.py**
- Flask web server
- Routes and API endpoints
- Loads and manages the prediction model
- Serves the web interface

### **src/predict.py**
- `FakeNewsPredictor` class
- Supports multiple model formats:
  - Content-based models (TF-IDF + linguistic features)
  - Notebook-trained models (vectorizer + classifier)
  - Legacy models (with preprocessing pipeline)
- Single and batch prediction capabilities
- Detailed article analysis with confidence scores

### **src/train.py**
- `FakeNewsClassifier` class
- Supports multiple algorithms:
  - Logistic Regression
  - Random Forest
  - SVM (Support Vector Machine)
  - Naive Bayes
  - Gradient Boosting
- Cross-validation and hyperparameter tuning
- Model persistence (save/load)

### **src/preprocess.py**
- `TextPreprocessor` class
- Text cleaning and normalization
- Tokenization, lemmatization, stemming
- Stopword removal
- URL, email, and special character handling

### **src/feature_extraction.py**
- `FeatureExtractor` class
- TF-IDF vectorization
- N-gram support (unigrams, bigrams)
- Configurable feature limits

### **src/content_features.py**
- Advanced linguistic pattern detection
- Content-based feature engineering
- Enhances model accuracy with linguistic signals

### **src/evaluate.py**
- Model performance metrics
- Confusion matrices
- ROC curves and AUC scores
- Classification reports

### **src/main.py**
- Training orchestration
- Interactive training interface
- Multi-model training and comparison

## 🚀 Usage Workflows

### **Running the Web Application**
```powershell
python app.py
```
Access at: http://127.0.0.1:5000

### **Training a New Model**
```powershell
cd src
python main.py
```

### **Making Predictions (CLI)**
```powershell
cd src
python predict.py --interactive
```

## 📊 Data Flow

```
User Input (Text Article)
    ↓
[preprocess.py] → Clean & normalize text
    ↓
[feature_extraction.py] → Extract TF-IDF features
[content_features.py] → Add linguistic features
    ↓
[predict.py] → Load model & predict
    ↓
Prediction Result + Confidence Score
    ↓
[app.py] → Display in web interface
```

## 🎓 Model Training Pipeline

```
Raw Dataset (CSV)
    ↓
[preprocess.py] → Clean & prepare text
    ↓
[feature_extraction.py] → Extract features
    ↓
[train.py] → Train classifier(s)
    ↓
[evaluate.py] → Evaluate performance
    ↓
[models/] → Save best model (.pkl)
```

## 🧹 Clean Code Practices Applied

✅ **Removed Redundancies**
- Eliminated duplicate training scripts
- Removed unused dataset creation files
- Cleaned up old model files

✅ **Modular Architecture**
- Clear separation of concerns
- Each module has a single responsibility
- Easy to test and maintain

✅ **Documentation**
- Comprehensive docstrings
- Clear README files
- Inline comments for complex logic

✅ **Organized Structure**
- Logical file organization
- Consistent naming conventions
- Clear directory hierarchy

## 🔧 Dependencies

Key packages (see requirements.txt for full list):
- **Flask**: Web framework
- **scikit-learn**: Machine learning
- **nltk**: Natural language processing
- **pandas**: Data manipulation
- **joblib**: Model serialization

## 📈 Current Model Performance

**Content-Based Logistic Regression** (Active Model)
- Test Accuracy: **99.7%**
- Cross-Validation Accuracy: **99.875%**
- Features: TF-IDF + Linguistic Patterns
- Dataset: balanced_2000_dataset.csv
