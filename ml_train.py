"""
ML Model Training Pipeline for Credit Default Prediction
Neostats AI Engineer Assignment
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, 
    confusion_matrix, classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# Paths
DATA_PATH = Path("data/application_train.csv")
MODEL_PATH = Path("models/credit_risk_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")

def load_and_prepare_data(filepath, test_size=0.3):
    """
    Load Home Credit dataset and prepare for training
    """
    print(f"📂 Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()} ({100*df.isnull().sum().sum()/(df.shape[0]*df.shape[1]):.2f}%)")
    print(f"Default rate: {df['TARGET'].sum() / len(df) * 100:.2f}%")
    
    # ===== DATA PREPROCESSING =====
    print("\n🔧 Preprocessing...")
    
    # Separate features and target
    X = df.drop('TARGET', axis=1)
    y = df['TARGET']
    
    # Handle missing values
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        X[col].fillna(X[col].median(), inplace=True)
    
    for col in categorical_cols:
        X[col].fillna(X[col].mode()[0] if X[col].mode().shape[0] > 0 else 'UNKNOWN', inplace=True)
    
    # Encode categorical variables
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    print(f"✅ Preprocessed shape: {X.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, X.columns


def handle_class_imbalance(X_train, y_train):
    """
    Handle class imbalance using SMOTE (Synthetic Minority Over-sampling)
    """
    print("\n⚖️ Handling class imbalance with SMOTE...")
    
    print(f"Before SMOTE - Class distribution:")
    print(f"  Non-default: {(y_train == 0).sum()}")
    print(f"  Default: {(y_train == 1).sum()}")
    print(f"  Ratio: 1:{(y_train == 0).sum() / (y_train == 1).sum():.1f}")
    
    smote = SMOTE(random_state=42, k_neighbors=5, n_jobs=-1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\nAfter SMOTE - Class distribution:")
    print(f"  Non-default: {(y_train_resampled == 0).sum()}")
    print(f"  Default: {(y_train_resampled == 1).sum()}")
    print(f"  Ratio: 1:{(y_train_resampled == 0).sum() / (y_train_resampled == 1).sum():.1f}")
    
    return X_train_resampled, y_train_resampled


def train_model(X_train, y_train):
    """
    Train XGBoost model with optimized hyperparameters
    """
    print("\n🤖 Training XGBoost model...")
    
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # Handle imbalance
        early_stopping_rounds=10
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )
    
    print("✅ Model trained successfully!")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance with multiple metrics
    """
    print("\n📊 Evaluating model...")
    
    # Predictions
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    y_train_pred = (y_train_pred_proba > 0.5).astype(int)
    y_test_pred = (y_test_pred_proba > 0.5).astype(int)
    
    # Metrics
    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    test_f1 = f1_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    
    print(f"\n{'='*50}")
    print("TRAINING SET METRICS")
    print(f"{'='*50}")
    print(f"ROC-AUC Score: {train_auc:.4f}")
    print(f"Accuracy: {(y_train_pred == y_train).sum() / len(y_train):.4f}")
    
    print(f"\n{'='*50}")
    print("TEST SET METRICS")
    print(f"{'='*50}")
    print(f"ROC-AUC Score: {test_auc:.4f} ⭐")
    print(f"F1-Score: {test_f1:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"Accuracy: {(y_test_pred == y_test).sum() / len(y_test):.4f}")
    
    print(f"\n{'='*50}")
    print("CONFUSION MATRIX")
    print(f"{'='*50}")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"True Negatives: {cm[0, 0]}")
    print(f"False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}")
    print(f"True Positives: {cm[1, 1]}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    return {
        'train_auc': train_auc,
        'test_auc': test_auc,
        'f1': test_f1,
        'precision': test_precision,
        'recall': test_recall,
        'y_test_pred_proba': y_test_pred_proba,
        'y_test': y_test
    }


def save_model(model, feature_names):
    """
    Save trained model and metadata
    """
    Path("models").mkdir(exist_ok=True)
    
    print(f"\n💾 Saving model to {MODEL_PATH}...")
    
    # Save model with metadata
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'feature_count': len(feature_names),
        'model_type': 'XGBClassifier',
        'threshold': 0.5  # Default decision threshold
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Model saved with {len(feature_names)} features")


def main():
    """
    Main training pipeline
    """
    print("="*60)
    print("CREDIT DEFAULT PREDICTION - MODEL TRAINING")
    print("Neostats AI Engineer Assignment")
    print("="*60)
    
    # Step 1: Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(
        DATA_PATH, test_size=0.3
    )
    
    # Step 2: Handle class imbalance
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train)
    
    # Step 3: Train model
    model = train_model(X_train_balanced, y_train_balanced)
    
    # Step 4: Evaluate
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Step 5: Save model
    save_model(model, feature_names)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"\nModel saved at: {MODEL_PATH}")
    print(f"Ready for deployment!")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
