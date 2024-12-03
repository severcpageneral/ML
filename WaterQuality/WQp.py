# Базовые библиотеки для работы с данными и вычислений
import numpy as np
import pandas as pd
import os
import time
import tempfile

# Библиотеки для машинного обучения
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Метрики и оценка модели
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Визуализация данных
import matplotlib.pyplot as plt
import seaborn as sns

# Инструменты для отслеживания экспериментов и сохранения моделей
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib

# Инициализация DagsHub (если используется)
import dagshub
dagshub.init(repo_owner='sever.cpa.general', repo_name='my-first-repo', mlflow=True)

# Загрузка данных
df = pd.read_csv('water_potability.csv')

# Обработка пропущенных значений (если есть)
# df.fillna(df.mean(), inplace=True)

# Проверка дисбаланса классов
print("Распределение классов в данных:")
print(df['Potability'].value_counts())

# Подготовка данных
X = df.drop(columns='Potability')
y = df['Potability']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Установка эксперимента MLflow
experiment_name = "Water Probability [RF]"
from mlflow.exceptions import MlflowException

try:
    experiment_id = mlflow.create_experiment(experiment_name)
except MlflowException:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# Запуск эксперимента в контекстном менеджере
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.set_tag("data_version", "v1.0")
    # Создание модели
    model = RandomForestClassifier(
        criterion='gini',
        n_estimators=30,
        max_depth=15,
        max_features=3,
        min_samples_split=5,
        random_state=33,
        class_weight='balanced'
    )

    # Логирование параметров модели
    mlflow.log_params(model.get_params())

    # Измерение времени обучения
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    mlflow.log_metric("training_time", training_time)

    # Измерение времени предсказания
    start_pred_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_pred_time
    mlflow.log_metric("prediction_time", prediction_time)

    # Вычисление метрик
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

    # Логирование метрик
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Вычисление и логирование ROC AUC
    y_score = model.predict_proba(X_test)
    if len(np.unique(y)) == 2:
        roc_auc = roc_auc_score(y_test, y_score[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr', average='weighted')
    mlflow.log_metric("roc_auc", roc_auc)

    # Вычисление отчета о классификации
    classification_report_text = classification_report(
        y_test,
        y_pred,
        zero_division=1
    )
    report = classification_report(
        y_test,
        y_pred,
        zero_division=1,
        output_dict=True
    )

    # Логирование отчета о классификации
    report_df = pd.DataFrame(report).transpose()
    temp_dir = tempfile.mkdtemp()
    report_path = os.path.join(temp_dir, "classification_report.csv")
    report_df.to_csv(report_path)
    mlflow.log_artifact(report_path, artifact_path="reports")

    # Логирование матрицы ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_path = os.path.join(temp_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path, artifact_path="plots")
    plt.close()  # Закрыть фигуру, чтобы освободить память

    # Логирование важности признаков
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feature_importances.sort_values(ascending=False, inplace=True)
    importance_path = os.path.join(temp_dir, "feature_importances.csv")
    feature_importances.to_csv(importance_path)
    mlflow.log_artifact(importance_path, artifact_path="feature_importances")

    # Логирование размера модели
    model_path = os.path.join(temp_dir, "model.joblib")
    joblib.dump(model, model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    mlflow.log_metric("model_size_mb", model_size)

    # Логирование модели с сигнатурой
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        signature=signature,
        registered_model_name="wq_rf_baseline",
    )

    # Вывод результатов
    print("\nМетрики модели:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nПодробный отчет:")
    print(classification_report_text)

    # Вывод распределения предсказаний
    print("\nРаспределение предсказаний:")
    print(pd.Series(y_pred).value_counts(normalize=True))