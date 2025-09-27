import argparse
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import your custom data processing functions from the src folder
from data_utils import load_data, prepare_xy, train_val_split


def build_pipeline():
    """Builds the scikit-learn preprocessing and modeling pipeline."""
    numeric_features = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
    categorical_features = ['Gender', 'Subscription Type', 'Contract Length']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='drop'
    )

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    return clf


def evaluate_model(model, X_val, y_val):
    """Calculates and returns a dictionary of performance metrics."""
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        'accuracy': float(accuracy_score(y_val, preds)),
        'precision': float(precision_score(y_val, preds, zero_division=0)),
        'recall': float(recall_score(y_val, preds, zero_division=0)),
        'f1': float(f1_score(y_val, preds, zero_division=0)),
    }
    if probs is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_val, probs))
        except Exception:
            metrics['roc_auc'] = None
    return metrics


def main(args):
    """Main function to run the training and logging workflow."""
    # Enable MLflow autologging. This automatically logs parameters, metrics, artifacts,
    # and the best model from the GridSearchCV.
    mlflow.autolog(log_models=True, exclusive=True)

    # Use a with statement to ensure the MLflow run is properly managed
    with mlflow.start_run():
        # Load and prepare data using the path from the command-line argument
        df = load_data(args.input_data)
        X, y = prepare_xy(df)
        X_train, X_val, y_train, y_val = train_val_split(X, y, test_size=0.2)

        pipeline = build_pipeline()

        # Define a small hyperparameter grid for demonstration
        param_grid = {
            'classifier__n_estimators': [50, 100, 150],
            'classifier__max_depth': [None, 10, 20],
        }
        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)

        # The best model is automatically logged by autologger, but we can
        # get it to evaluate on the validation set.
        best_model = grid.best_estimator_
        
        # Evaluate the best model on the held-out validation set
        validation_metrics = evaluate_model(best_model, X_val, y_val)
        print("Best parameters found:", grid.best_params_)
        print("Validation metrics:", validation_metrics)

        # Explicitly log validation metrics to MLflow for easy tracking
        mlflow.log_metrics({f"validation_{k}": v for k, v in validation_metrics.items() if v is not None})


if __name__ == "__main__":
    # Setup the argument parser to accept the data path from Azure ML
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data",
        type=str,
        help="Path to the input data CSV file",
        required=True
    )
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)