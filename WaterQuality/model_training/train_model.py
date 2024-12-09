# WaterQuality/model_training/train_model.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
import tensorflow as tf
from tensorflow import keras
import time
from create_model import create_model

def train_and_evaluate_model(X, y, model_config, hyperparams):
    """
    Обучает модель, используя параметры из конфигурационного файла
    """
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=hyperparams['train_test_split']['test_size'],
        random_state=hyperparams['train_test_split']['random_state']
    )

    # Стандартизация признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Создание модели
    model = create_model(input_dim=X.shape[1], model_config=model_config)

    # Настройка ранней остановки для предотвращения переобучения
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=hyperparams['early_stopping']['monitor'],
        patience=hyperparams['early_stopping']['patience'],
        restore_best_weights=hyperparams['early_stopping']['restore_best_weights']
    )

    # Замер времени обучения
    start_time = time.time()
    history = model.fit(
        X_train_scaled, y_train,
        epochs=hyperparams['epochs'],
        batch_size=hyperparams['batch_size'],
        validation_split=hyperparams['validation_split'],
        callbacks=[early_stopping],
        verbose=2
    )
    training_time = time.time() - start_time

    # Получение предсказаний
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    y_pred_proba = model.predict(X_test_scaled)

    # Расчет метрик качества модели
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=1),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=1),
        "f1": f1_score(y_test, y_pred, average='weighted', zero_division=1),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "training_time": training_time,
    }

    return {
        'model': model,
        'scaler': scaler,
        'history': history,
        'metrics': metrics,
        'predictions': {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        },
        'data': {
            'X_test_scaled': X_test_scaled
        }
    }