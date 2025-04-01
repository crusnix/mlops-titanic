import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# Load dataset
data = pd.read_csv("data/raw/titanic.csv")
data = data.dropna(subset=["Age", "Fare", "Pclass", "Survived"])  # Drop missing values

# Features & target
X = data[["Pclass", "Age", "Fare"]]
y = data["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
