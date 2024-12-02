import numpy as np
import pandas as pd
import os

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier


import dagshub
dagshub.init(repo_owner='sever.cpa.general', repo_name='my-first-repo', mlflow=True)

# Установка URI для tracking server
# mlflow.set_tracking_uri("http://localhost:5000")  # измените на ваш URI

# Установка директории для сохранения модели
#current_dir = os.path.dirname(os.path.abspath(__file__))
#model_dir = os.path.join(current_dir, 'model')

# Создаём директорию, если её нет
#if not os.path.exists(model_dir):
 #   os.makedirs(model_dir)

# Путь для сохранения модели
#model_path = os.path.join(model_dir, 'water_quality_model.pkl')

# Загрузка данных
df = pd.read_csv('water_potability.csv')

# Подготовка данных
X = df.drop(columns='Potability')
y = df['Potability']

# Проверка баланса классов
print("\nРаспределение классов в данных:")
print(pd.Series(y).value_counts(normalize=True))

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Нормализация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Вычисление весов классов
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Установка эксперимента MLflow
experiment_name = "Water Probability [RF]"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
except:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# Запуск эксперимента в контекстном менеджере
try:
    with mlflow.start_run(experiment_id=experiment_id):
        # Создание и обучение модели RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, class_weight=class_weight_dict, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Предсказание и вычисление метрик
        y_pred = model.predict(X_test_scaled)

        # Вычисление метрик
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            zero_division=1,
            output_dict=True
        )

        # Infer the model signature
        y_pred = model.predict(X_test)
        signature = infer_signature(X_test, y_pred)

        # Логирование параметров и метрик
        mlflow.log_params({
            "n_estimators": 100,
            "model_architecture": "RandomForestClassifier"
        })

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])

        # Log the sklearn model and register as version 1
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            signature=signature,
            registered_model_name="model_random_forest",
        )

        # Логирование отчета классификации как текстового артефакта
        classification_report_text = classification_report(y_test, y_pred, zero_division=1)
        #with open(os.path.join(model_dir, "classification_report.txt"), "w") as f:
            #f.write(classification_report_text)
        #mlflow.log_artifact(os.path.join(model_dir, "classification_report.txt"), artifact_path="metrics")

        # Визуализация важности признаков
        feature_importances = model.feature_importances_
        features = X.columns
        plt.figure(figsize=(10, 6))
        plt.barh(features, feature_importances, color='skyblue')
        plt.xlabel('Важность признака')
        plt.title('Важность признаков для модели RandomForest')
        plt.tight_layout()
        #feature_importance_path = os.path.join(model_dir, "feature_importance.png")
        #plt.savefig(feature_importance_path)
        #mlflow.log_artifact(feature_importance_path, artifact_path="plots")

        # Сохранение модели
        import joblib
        #joblib.dump(model, model_path)
        #mlflow.log_artifact(model_path, artifact_path="model")

        # Вывод результатов
        print("\nМетрики модели:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nПодробный отчет:")
        print(classification_report_text)
        #print(f"\nМодель сохранена в: {model_path}")

        # Вывод распределения предсказаний
        print("\nРаспределение предсказаний:")
        print(pd.Series(y_pred.flatten()).value_counts(normalize=True))
finally:
    mlflow.end_run()
