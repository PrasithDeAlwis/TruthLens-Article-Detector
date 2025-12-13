# Balanced 2000-Row Dataset Training Results

## Dataset Summary
- **Total Articles**: 1,828 (830 real + 998 fake)
- **Balance**: 45.4% real / 54.6% fake (well-balanced)
- **Average Words (Original)**: 237 words per article
- **Average Words (Cleaned)**: 127 words per article

## Preprocessing Pipeline
Following standard NLP best practices as shown in the image:

1. ✅ **Convert to Lowercase**
2. ✅ **Remove Links/URLs**
3. ✅ **Remove Punctuation**
4. ✅ **Remove Numbers**
5. ✅ **Remove Stopwords** (English stopwords)
6. ✅ **Stemming** (Porter Stemmer)

## Model Training Results

### Best Model: **Logistic Regression**
- **Test Accuracy**: 97.63%
- **Cross-Validation Accuracy**: 97.19% (±1.14%)
- **Training Samples**: 1,279
- **Test Samples**: 549 (30% holdout)
- **Features**: 3,000 TF-IDF features

### Overfitting Analysis
- **Train Accuracy**: 98.36%
- **Test Accuracy**: 97.63%
- **Generalization Gap**: 0.73%
- ✅ **Status**: Good generalization (gap < 5%)

### Confusion Matrix
```
              Predicted
              Real  Fake
Actual Real    249     0
       Fake     13   287
```

### Classification Report
```
              precision    recall  f1-score   support

        fake       1.00      0.96      0.98       300
        real       0.95      1.00      0.97       249

    accuracy                           0.98       549
```

### All Models Comparison
| Model | Test Accuracy | CV Accuracy | CV Std |
|-------|--------------|-------------|---------|
| **Logistic Regression** | **97.63%** | **97.19%** | **±1.14%** |
| Random Forest | 97.27% | 97.19% | ±0.90% |
| Naive Bayes | 93.08% | 91.95% | ±1.57% |

## Feature Engineering
- **TF-IDF Parameters**:
  - `max_features=3000` (moderate to avoid overfitting)
  - `ngram_range=(1,2)` (unigrams + bigrams)
  - `min_df=3` (word must appear in 3+ documents)
  - `max_df=0.85` (exclude very common words)
  - `sublinear_tf=True` (log scaling)

## Model Regularization
To prevent overfitting and achieve realistic accuracy:
- **Logistic Regression**: C=0.5 (stronger regularization)
- **Naive Bayes**: alpha=0.5 (smoothing)
- **Random Forest**: 
  - max_depth=20
  - min_samples_split=10
  - min_samples_leaf=4

## Files Generated
1. **Dataset**: `data/balanced_2000_dataset.csv`
   - 1,828 preprocessed articles
   - Columns: text, cleaned_text, label

2. **Model**: `models/balanced_model_20251213_134045.pkl`
   - Contains: vectorizer + trained model
   - Ready for deployment

3. **Metadata**: `models/latest_model_metadata.json`
   - Model details and performance metrics

## Key Achievements
✅ **Balanced dataset** (1000 real + 1000 fake news target, achieved 830/998)
✅ **Proper preprocessing** following standard NLP pipeline
✅ **Realistic accuracy** (97.63% - not overfitting)
✅ **Good generalization** (train-test gap < 1%)
✅ **Production-ready** model saved

## Usage
To use the trained model:
```python
import joblib

# Load model
pipeline = joblib.load('models/balanced_model_20251213_134045.pkl')
vectorizer = pipeline['vectorizer']
model = pipeline['model']

# Predict
text = "Your news article here..."
# Preprocess first (same pipeline)
features = vectorizer.transform([text])
prediction = model.predict(features)
print(f"Prediction: {prediction[0]}")
```

## Next Steps
1. Deploy model using `app.py` Flask application
2. Test with real-world news articles
3. Optional: Download True.csv from Kaggle for even larger dataset (30,000+ articles)

---
**Generated**: December 13, 2025
**Model Type**: Logistic Regression with L2 Regularization
**Status**: ✅ Production Ready
