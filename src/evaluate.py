"""
Model Evaluation Module
Provides comprehensive evaluation metrics and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
import os


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization
    """
    
    def __init__(self, model, X_test, y_test, label_names=None):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            label_names (list): Names of labels
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            self.y_proba = model.predict_proba(X_test)
        else:
            self.y_proba = None
        
        self.label_names = label_names if label_names else [str(i) for i in np.unique(y_test)]
    
    def calculate_metrics(self):
        """
        Calculate comprehensive metrics
        
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred, average='weighted'),
            'recall': recall_score(self.y_test, self.y_pred, average='weighted'),
            'f1_score': f1_score(self.y_test, self.y_pred, average='weighted')
        }
        
        return metrics
    
    def print_classification_report(self):
        """Print detailed classification report"""
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(self.y_test, self.y_pred, target_names=self.label_names))
    
    def plot_confusion_matrix(self, save_path=None, figsize=(8, 6)):
        """
        Plot confusion matrix
        
        Args:
            save_path (str): Path to save plot
            figsize (tuple): Figure size
        """
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names,
                   yticklabels=self.label_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, save_path=None, figsize=(8, 6)):
        """
        Plot ROC curve (for binary classification)
        
        Args:
            save_path (str): Path to save plot
            figsize (tuple): Figure size
        """
        if self.y_proba is None:
            print("Model does not support probability predictions")
            return
        
        if len(self.label_names) != 2:
            print("ROC curve is only available for binary classification")
            return
        
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, save_path=None, figsize=(8, 6)):
        """
        Plot precision-recall curve (for binary classification)
        
        Args:
            save_path (str): Path to save plot
            figsize (tuple): Figure size
        """
        if self.y_proba is None:
            print("Model does not support probability predictions")
            return
        
        if len(self.label_names) != 2:
            print("Precision-recall curve is only available for binary classification")
            return
        
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_proba[:, 1])
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-recall curve saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names, top_n=20, save_path=None, figsize=(10, 8)):
        """
        Plot feature importance (for tree-based models)
        
        Args:
            feature_names (list): Names of features
            top_n (int): Number of top features to show
            save_path (str): Path to save plot
            figsize (tuple): Figure size
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("Model does not have feature importances")
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=figsize)
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def generate_full_report(self, output_dir='../reports'):
        """
        Generate comprehensive evaluation report with all plots
        
        Args:
            output_dir (str): Directory to save reports
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate and print metrics
        metrics = self.calculate_metrics()
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        # Print classification report
        self.print_classification_report()
        
        # Generate plots
        self.plot_confusion_matrix(save_path=os.path.join(output_dir, 'confusion_matrix.png'))
        
        if len(self.label_names) == 2 and self.y_proba is not None:
            self.plot_roc_curve(save_path=os.path.join(output_dir, 'roc_curve.png'))
            self.plot_precision_recall_curve(save_path=os.path.join(output_dir, 'precision_recall_curve.png'))
        
        print(f"\nReports saved to {output_dir}")


def compare_models(results_dict, save_path=None):
    """
    Compare multiple models and visualize results
    
    Args:
        results_dict (dict): Dictionary of model results
        save_path (str): Path to save comparison plot
    """
    model_names = list(results_dict.keys())
    accuracies = [results_dict[name]['accuracy'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color='steelblue')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("Use this module with a trained model to generate comprehensive evaluation reports")
