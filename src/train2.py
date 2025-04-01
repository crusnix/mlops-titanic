import logging
import os
from pathlib import Path
import pickle
import joblib

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

# ---------------------------
# Setup Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ---------------------------
# Dummy/Helper Functions
# ---------------------------
def start_mlflow_server():
    """
    Stub function: Implement a mechanism to start the MLflow server if needed.
    For example, you might launch it using subprocess.
    """
    logging.info("Starting MLflow server... (stub)")
    # Example: subprocess.Popen(["mlflow", "ui"]) if desired
    pass

def load_dataframe(path):
    """Load a CSV into a DataFrame."""
    logging.info(f"Loading dataframe from {path}")
    return pd.read_csv(path)

def make_X_y(df, target):
    """Split the dataframe into features and target."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def get_predictions(model, X):
    """Return model predictions."""
    return model.predict(X)

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return accuracy, f1, precision, recall

def evaluate_and_log(model, X, y, dataset_name):
    """Evaluate the model and log metrics."""
    y_pred = get_predictions(model, X)
    accuracy, f1, precision, recall = calculate_metrics(y, y_pred)
    
    logging.info(f'\nMetrics for {dataset_name} dataset:')
    logging.info(f'Accuracy: {accuracy:.4f}')
    logging.info(f'F1 Score: {f1:.4f}')
    logging.info(f'Precision: {precision:.4f}')
    logging.info(f'Recall: {recall:.4f}')

# ---------------------------
# MLflow Setup
# ---------------------------
def setup_mlflow():
    start_mlflow_server()
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment_name = "Hyperparameter Tuning"
    mlflow.set_experiment(experiment_name)
    logging.info(f"MLflow tracking URI set to http://localhost:5000 and experiment name set to {experiment_name}")

# ---------------------------
# Define Models & Hyperparameter Grids
# ---------------------------
models_to_tune = {
    'RandomForest': (
        RandomForestClassifier(),
        {
            'n_estimators': [500],
            'max_depth': [10, None],
            'min_samples_split': [10],
            'min_samples_leaf': [1],
            'max_features': ['sqrt']
        }
    ),
    'GradientBoosting': (
        GradientBoostingClassifier(),
        {
            'n_estimators': [400],
            'learning_rate': [0.1],
            'max_depth': [4],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['sqrt']
        }
    )
}

# ---------------------------
# Hyperparameter Tuning Function
# ---------------------------
def hyperparameter_tuning(model, param_dist, X_train, y_train, X_val, y_val, n_iter=100, cv=5):
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2,
        random_state=9
    )
    random_search.fit(X_train, y_train)
    # Optionally, you can evaluate on validation set here if needed.
    return random_search.best_estimator_, random_search.best_score_, random_search.best_params_

# ---------------------------
# Main Execution Flow
# ---------------------------
def main():
    # Setup MLflow tracking
    setup_mlflow()
    
    # Define file paths and target variable
    root_path = Path(__file__).parent
    train_path = root_path / 'data' / 'train.csv'
    val_path = root_path / 'data' / 'validation.csv'
    TARGET = 'Survived'
    
    # Load training data
    train_data = load_dataframe(train_path)
    X_train, y_train = make_X_y(train_data, TARGET)
    
    # Load validation data
    val_data = load_dataframe(val_path)
    X_val, y_val = make_X_y(val_data, TARGET)
    
    tuned_model_results = []
    
    # Hyperparameter tuning for each model
    for model_name, (model, param_grid) in models_to_tune.items():
        logging.info(f"Starting hyperparameter tuning for {model_name}...")
        best_model, best_score, best_params = hyperparameter_tuning(model, param_grid, X_train, y_train, X_val, y_val)
        logging.info(f"Best score for {model_name}: {best_score:.4f}")
        logging.info(f"Best parameters for {model_name}: {best_params}")
        
        # Log best model to MLflow (include model signature)
        signature = infer_signature(X_train, best_model.predict(X_train))
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(best_params)
            mlflow.log_metric("best_score", best_score)
            mlflow.sklearn.log_model(best_model, f"{model_name}_best", signature=signature)
        
        tuned_model_results.append((model_name, best_score, best_model))
    
    # Sort tuned models based on score (descending order)
    tuned_model_results.sort(key=lambda x: x[1], reverse=True)
    best_tuned_model1 = tuned_model_results[0][2] if len(tuned_model_results) > 0 else None
    best_tuned_model2 = tuned_model_results[1][2] if len(tuned_model_results) > 1 else None
    
    # Save the best tuned model locally using joblib
    if best_tuned_model1 is not None:
        model_filename = tuned_model_results[0][0] + "_tuned.joblib"
        tuned_models_dir = root_path / 'models' / 'tuned_models'
        tuned_models_dir.mkdir(parents=True, exist_ok=True)
        model_path = tuned_models_dir / model_filename
        joblib.dump(best_tuned_model1, model_path)
        logging.info(f"Best tuned model saved at {model_path}")
    
    # Load preprocessor if it exists
    preprocessor_path = root_path / 'models' / 'transformers' / 'preprocessor.joblib'
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)
        logging.info(f"Preprocessor loaded from {preprocessor_path}")
    else:
        preprocessor = None
        logging.info("No preprocessor found.")
    
    # Evaluate the best tuned model on the validation set
    if best_tuned_model1 is not None:
        evaluate_and_log(best_tuned_model1, X_val, y_val, "Validation")

# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    main()
