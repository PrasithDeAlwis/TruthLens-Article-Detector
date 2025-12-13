import sys
import os
sys.path.append('src')

from predict import FakeNewsPredictor
import glob

# Load the latest content-based model
model_files = glob.glob('models/content_based_model_*.pkl')
if model_files:
    model_path = max(model_files, key=lambda x: x)
    print(f"Using content-based model: {model_path}\n")
else:
    model_path = 'models/notebook_trained_model_20251213_141132.pkl'
    print(f"Using notebook model: {model_path}\n")

predictor = FakeNewsPredictor(model_path)

print("=" * 70)
print("TESTING MODEL PREDICTIONS")
print("=" * 70)

# Test cases
test_cases = [
    {
        "text": "The president announced new economic policies today in Washington.",
        "expected": "real"
    },
    {
        "text": "Scientists discovered water on Mars after years of research.",
        "expected": "real"
    },
    {
        "text": "BREAKING: Obama is secretly a reptilian alien from outer space! Click here to learn the SHOCKING truth that the mainstream media doesn't want you to know!",
        "expected": "fake"
    },
    {
        "text": "You won't believe what this celebrity did! Doctors hate this one weird trick!",
        "expected": "fake"
    },
    {
        "text": "The stock market closed higher today.",
        "expected": "real"
    }
]

print("\n" + "=" * 70)
for i, test in enumerate(test_cases, 1):
    print(f"\nTest {i}:")
    print(f"Text: {test['text']}")
    print(f"Expected: {test['expected']}")
    
    analysis = predictor.analyze_article(test['text'])
    
    print(f"Predicted: {analysis['prediction']}")
    print(f"Confidence: {analysis['confidence']*100:.2f}%")
    print(f"Probabilities: Real={analysis['probabilities'].get('real', 0)*100:.1f}%, Fake={analysis['probabilities'].get('fake', 0)*100:.1f}%")
    
    if analysis['prediction'] == test['expected']:
        print("✓ CORRECT")
    else:
        print("✗ INCORRECT")
    print("-" * 70)
