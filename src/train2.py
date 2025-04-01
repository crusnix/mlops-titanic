import mlflow
import mlflow.sklearn
import pandas as pd
import logging
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import yaml

# Load dataset
data = pd.read_csv("data/raw/titanic.csv")
data = data.dropna(subset=["Age", "Fare", "Pclass", "Survived"])  # Drop missing values

# Features & target
X = data[["Pclass", "Age", "Fare"]]
y = data["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ),
    'LogisticRegression': LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    'DecisionTree': DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=5
    )
}


for model_name, model in models.items():
    logging.info(f'{model_name} is training...')
    
    # Train the model
    trained_model = train_model(model=model, X_train=X_train, y_train=y_train)
    
    # Evaluate the model
    accuracy, f1, precision, recall = evaluate_model(model=trained_model, X_test=X_test, y_test=y_test)
    
    # Save the model
    save_path = f"models/{model_name}.pkl"
    save_model(model=trained_model, save_path=save_path)
    
    logging.info(f'{model_name} training completed and saved at {save_path}')

# MLflow Experiment
mlflow.set_experiment("Titanic_MLOps")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "titanic_model")
    print(f"Model trained with accuracy: {acc:.4f}")
