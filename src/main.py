"""
Main Training Script
Complete workflow for training a fake news detection model
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import FakeNewsClassifier, train_multiple_models
from evaluate import ModelEvaluator, compare_models
from create_sample_data import save_sample_dataset


def main():
    """
    Main training workflow
    """
    print("=" * 70)
    print("TRUTHLENS FAKE NEWS DETECTION - TRAINING WORKFLOW")
    print("=" * 70)
    
    # Check if data exists
    data_path = '../data/sample_news.csv'
    
    if not os.path.exists(data_path):
        print("\nNo dataset found. Creating sample dataset...")
        data_path = save_sample_dataset()
        print("\nNOTE: This is a small sample dataset for demonstration.")
        print("For real-world use, please provide a larger dataset with")
        print("'text' and 'label' columns in CSV format.")
        print("\nYou can use datasets like:")
        print("- LIAR dataset")
        print("- ISOT Fake News Dataset")
        print("- Kaggle Fake News Detection datasets")
    
    # Ask user which approach to use
    print("\n" + "=" * 70)
    print("TRAINING OPTIONS")
    print("=" * 70)
    print("1. Train single model (faster)")
    print("2. Train and compare multiple models (recommended)")
    
    choice = input("\nSelect option (1 or 2): ").strip()
    
    if choice == '1':
        # Single model training
        print("\nAvailable models:")
        print("1. Logistic Regression")
        print("2. Random Forest")
        print("3. Naive Bayes")
        print("4. Support Vector Machine (SVM)")
        
        model_choice = input("\nSelect model (1-4): ").strip()
        model_types = {
            '1': 'logistic_regression',
            '2': 'random_forest',
            '3': 'naive_bayes',
            '4': 'svm'
        }
        
        model_type = model_types.get(model_choice, 'logistic_regression')
        
        # Train model
        classifier = FakeNewsClassifier(model_type=model_type)
        X_train, X_test, y_train, y_test = classifier.prepare_data(data_path)
        classifier.train(X_train, y_train)
        results = classifier.evaluate(X_test, y_test)
        
        # Save model
        model_path = classifier.save()
        
        # Generate evaluation report
        label_names = {v: k for k, v in classifier.label_mapping.items()}
        evaluator = ModelEvaluator(
            classifier.model, X_test, y_test,
            label_names=[label_names[i] for i in sorted(label_names.keys())]
        )
        evaluator.generate_full_report()
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Model saved to: {model_path}")
        print(f"Test Accuracy: {results['accuracy']:.4f}")
    
    elif choice == '2':
        # Multiple models training
        results, best_model_type = train_multiple_models(data_path)
        
        # Save best model
        if 'model' in results[best_model_type]:
            best_classifier = results[best_model_type]['model']
            model_path = best_classifier.save()
            
            print("\n" + "=" * 70)
            print("TRAINING COMPLETE!")
            print("=" * 70)
            print(f"Best model: {best_model_type}")
            print(f"Best accuracy: {results[best_model_type]['accuracy']:.4f}")
            print(f"Model saved to: {model_path}")
        
        # Compare models visually
        compare_models(results, save_path='../reports/model_comparison.png')
    
    else:
        print("Invalid choice. Exiting.")
        return
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. To make predictions, run: python predict.py --interactive")
    print("2. To predict from file: python predict.py --model <path> --file <data.csv>")
    print("3. Check the 'reports' folder for evaluation plots")
    print("=" * 70)


if __name__ == "__main__":
    main()
