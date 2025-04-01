import mlflow
import mlflow.sklearn

def register_best_model(best_run_id, artifact_name, model_name):
    # Формирование URI модели
    model_uri = f"runs:/{best_run_id}/{artifact_name}"
    
    # Регистрация модели в MLflow Model Registry
    model_details = mlflow.register_model(model_uri, model_name)
    print(f"Модель зарегистрирована. Run ID: {best_run_id}, версия: {model_details.version}")
    
    # Перевод модели в стадию Production и архивирование предыдущих версий
    mlflow.transition_model_version_stage(
        name=model_name,
        version=model_details.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Модель {model_name} версии {model_details.version} переведена в Production.")

if __name__ == "__main__":
    # Пример вызова функции регистрации.
    # Эти параметры должны быть получены из процесса обучения или переданы через аргументы.
    best_run_id = "your_best_run_id"         # ID лучшего run, полученный после обучения
    artifact_name = "gender_model"            # Имя артефакта, как он сохранён в MLflow
    model_name = "Titanic_Gender_Model"         # Имя модели для регистрации
    
    register_best_model(best_run_id, artifact_name, model_name)
