import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier  # Убедитесь, что установлен xgboost

# Загрузка данных и подготовка
data_gender = pd.read_csv("data/raw/titanic.csv")
data_gender = data_gender.dropna(subset=["Age", "Fare", "Pclass", "Sex"])
# Преобразуем признак Sex в числовой: male -> 1, female -> 0
data_gender["Sex_encoded"] = data_gender["Sex"].apply(lambda x: 1 if x.lower() == "male" else 0)

# Определяем признаки и целевую переменную для классификации по полу
X_gender = data_gender[["Pclass", "Age", "Fare", "Survived"]]
y_gender = data_gender["Sex_encoded"]

X_train_gender, X_test_gender, y_train_gender, y_test_gender = train_test_split(
    X_gender, y_gender, test_size=0.2, random_state=42
)

# Устанавливаем эксперимент в MLflow для модели пола
mlflow.set_experiment("Titanic_Gender")

# --- Модель 1: LogisticRegression ---
with mlflow.start_run(run_name="Gender_Model_LogReg"):
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X_train_gender, y_train_gender)
    predictions_lr = model_lr.predict(X_test_gender)
    
    acc_lr = accuracy_score(y_test_gender, predictions_lr)
    precision_lr = precision_score(y_test_gender, predictions_lr)
    recall_lr = recall_score(y_test_gender, predictions_lr)
    f1_lr = f1_score(y_test_gender, predictions_lr)
    
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", acc_lr)
    mlflow.log_metric("precision", precision_lr)
    mlflow.log_metric("recall", recall_lr)
    mlflow.log_metric("f1_score", f1_lr)
    mlflow.sklearn.log_model(model_lr, "logistic_regression_gender_model")
    
    print(f"LogisticRegression trained with accuracy: {acc_lr:.4f}")

# --- Модель 2: RandomForestClassifier ---
with mlflow.start_run(run_name="Gender_Model_RandomForest"):
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train_gender, y_train_gender)
    predictions_rf = model_rf.predict(X_test_gender)
    
    acc_rf = accuracy_score(y_test_gender, predictions_rf)
    precision_rf = precision_score(y_test_gender, predictions_rf)
    recall_rf = recall_score(y_test_gender, predictions_rf)
    f1_rf = f1_score(y_test_gender, predictions_rf)
    
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc_rf)
    mlflow.log_metric("precision", precision_rf)
    mlflow.log_metric("recall", recall_rf)
    mlflow.log_metric("f1_score", f1_rf)
    mlflow.sklearn.log_model(model_rf, "random_forest_gender_model")
    
    print(f"RandomForestClassifier trained with accuracy: {acc_rf:.4f}")

# --- Модель 3: XGBClassifier ---
with mlflow.start_run(run_name="Gender_Model_XGBoost"):
    model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model_xgb.fit(X_train_gender, y_train_gender)
    predictions_xgb = model_xgb.predict(X_test_gender)
    
    acc_xgb = accuracy_score(y_test_gender, predictions_xgb)
    precision_xgb = precision_score(y_test_gender, predictions_xgb)
    recall_xgb = recall_score(y_test_gender, predictions_xgb)
    f1_xgb = f1_score(y_test_gender, predictions_xgb)
    
    mlflow.log_param("model", "XGBClassifier")
    mlflow.log_metric("accuracy", acc_xgb)
    mlflow.log_metric("precision", precision_xgb)
    mlflow.log_metric("recall", recall_xgb)
    mlflow.log_metric("f1_score", f1_xgb)
    mlflow.sklearn.log_model(model_xgb, "xgboost_gender_model")
    
    print(f"XGBClassifier trained with accuracy: {acc_xgb:.4f}")
