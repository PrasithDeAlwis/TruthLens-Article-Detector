# TruthLens - Coding Standards & Best Practices

This document outlines the programming best practices and clean code principles applied in the TruthLens project.

## 🎯 Design Principles

### 1. **Single Responsibility Principle (SRP)**
Each module has a single, well-defined purpose:
- `predict.py` - Only handles predictions
- `train.py` - Only handles model training
- `preprocess.py` - Only handles text preprocessing
- `feature_extraction.py` - Only handles feature engineering
- `evaluate.py` - Only handles model evaluation

### 2. **DRY (Don't Repeat Yourself)**
- Common functionality is centralized in reusable modules
- No code duplication across files
- Shared utilities are properly imported

### 3. **Modularity**
- Clear separation of concerns
- Easy to test individual components
- Simple to add new features or models

### 4. **Extensibility**
- Easy to add new classification algorithms
- Support for multiple model formats
- Configurable preprocessing pipeline

## 📝 Code Organization

### **Directory Structure**
```
✓ Clear hierarchy
✓ Logical grouping of related files
✓ Separation of source code, data, models, and web assets
```

### **File Naming**
```
✓ Descriptive names (predict.py, not p.py)
✓ Lowercase with underscores (feature_extraction.py)
✓ Consistent conventions across project
```

### **Module Structure**
Each Python module follows this pattern:
1. Module docstring (purpose, author, version)
2. Imports (standard library, third-party, local)
3. Constants and configuration
4. Class definitions
5. Function definitions
6. Main execution block (if applicable)

## 📖 Documentation Standards

### **Module Docstrings**
```python
"""
Flask Web Application for TruthLens Fake News Detection

This module provides a web interface for the TruthLens fake news detection system.
It loads trained machine learning models and provides real-time predictions through
a user-friendly web interface.

Author: TruthLens Team
Version: 1.0
"""
```

### **Function Docstrings**
```python
def load_model():
    """
    Load the best available trained model.
    
    Priority order:
    1. Content-based model (best performance - 99.7% accuracy)
    2. Most recently created model file
    
    Returns:
        str: Name of the loaded model file, or None if no model found
    """
```

### **Class Docstrings**
```python
class FakeNewsPredictor:
    """
    Make predictions on new articles using trained models.
    
    Supports multiple model formats:
    - Content-based models (TF-IDF + linguistic features)
    - Notebook-trained models (vectorizer + classifier)
    - Legacy models (with preprocessing pipeline)
    """
```

## ✅ Code Quality Practices

### **1. Error Handling**
```python
try:
    predictor = FakeNewsPredictor(model_path)
    return os.path.basename(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    return None
```

### **2. Input Validation**
```python
if not article_text:
    return jsonify({
        'error': 'Please enter article text',
        'success': False
    })
```

### **3. Type Hints** (where applicable)
```python
def predict_single(self, text: str, return_proba: bool = False) -> Union[str, Tuple]:
    """Predict label for a single article"""
```

### **4. Constants**
```python
# Instead of magic numbers
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
TEST_SIZE = 0.2
```

### **5. Meaningful Variable Names**
```python
# Good ✓
article_text = request.form.get('article', '').strip()
prediction_confidence = analysis['confidence']

# Bad ✗
txt = request.form.get('article', '').strip()
conf = analysis['confidence']
```

## 🧪 Testing Considerations

### **Testability**
- Functions have clear inputs and outputs
- Minimal side effects
- Dependencies can be mocked

### **Example Test Structure**
```python
def test_load_model():
    """Test model loading functionality"""
    model_name = load_model()
    assert model_name is not None
    assert predictor is not None
```

## 🔒 Security Best Practices

### **1. Input Sanitization**
- All user input is validated
- Text is stripped of dangerous characters
- Length limits enforced

### **2. Error Messages**
- No sensitive information in error messages
- Generic messages for security issues
- Detailed logging for debugging

### **3. Configuration**
```python
# Secrets should be in environment variables, not hardcoded
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-dev-key')
```

## 📊 Performance Optimization

### **1. Model Loading**
- Model loaded once at startup (not per request)
- Global predictor instance for reuse

### **2. Efficient Data Processing**
- Batch predictions when possible
- Vectorized operations using numpy/pandas

### **3. Resource Management**
```python
# Use context managers
with open(file_path, 'r') as f:
    data = f.read()
```

## 🎨 Code Style

### **PEP 8 Compliance**
- 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Two blank lines between functions/classes
- One blank line between methods

### **Import Organization**
```python
# 1. Standard library imports
import os
import sys

# 2. Third-party imports
import pandas as pd
from flask import Flask, jsonify

# 3. Local application imports
from predict import FakeNewsPredictor
```

### **Naming Conventions**
- Classes: `PascalCase` (e.g., `FakeNewsPredictor`)
- Functions/Methods: `snake_case` (e.g., `load_model`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_FEATURES`)
- Private members: `_leading_underscore` (e.g., `_initialize_model`)

## 🔄 Version Control Best Practices

### **.gitignore**
```
__pycache__/
*.pyc
.venv/
venv/
*.pkl  # Large model files
*.log
.env   # Environment variables
```

### **Commit Messages**
```
✓ Clear and descriptive
✓ Present tense ("Add feature" not "Added feature")
✓ Reference issue numbers when applicable
```

## 📦 Dependency Management

### **requirements.txt**
- Pinned versions for reproducibility
- Only necessary dependencies
- Clear separation of dev and production dependencies

```txt
Flask==3.0.0
scikit-learn==1.3.2
pandas==2.1.4
nltk==3.8.1
```

## 🚀 Deployment Readiness

### **Environment Configuration**
- Development vs Production settings
- Environment variables for secrets
- Debug mode disabled in production

### **Logging**
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Model loaded successfully")
logger.error(f"Error: {e}")
```

## 📈 Continuous Improvement

### **Code Review Checklist**
- [ ] Code follows style guide
- [ ] Functions have docstrings
- [ ] Error handling implemented
- [ ] No code duplication
- [ ] Tests written (if applicable)
- [ ] Documentation updated
- [ ] No hardcoded secrets

### **Refactoring Priorities**
1. Remove code duplication
2. Improve naming clarity
3. Add missing documentation
4. Enhance error handling
5. Optimize performance bottlenecks

## 🎓 Learning Resources

- **PEP 8**: Python Style Guide
- **Clean Code** by Robert C. Martin
- **Effective Python** by Brett Slatkin
- **Flask Best Practices**: Official documentation

---

**Last Updated**: December 15, 2025  
**Maintained by**: TruthLens Development Team
