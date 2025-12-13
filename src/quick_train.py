"""Quick model training for web app demo"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import FakeNewsClassifier

# Train a simple logistic regression model
print("Training model for web app...")
classifier = FakeNewsClassifier(model_type='logistic_regression')

# Use the sample data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'sample_news.csv')
X_train, X_test, y_train, y_test = classifier.prepare_data(data_path)

# Train
classifier.train(X_train, y_train, cv_folds=2)  # Fast training

# Evaluate
results = classifier.evaluate(X_test, y_test)

# Save
model_dir = os.path.join(script_dir, '..', 'models')
model_path = classifier.save(model_dir)

print("\n" + "=" * 60)
print("Model trained and saved successfully!")
print(f"Test Accuracy: {results['accuracy']:.2%}")
print(f"Model saved to: {model_path}")
print("\nYou can now run the web app with: python app.py")
print("=" * 60)
