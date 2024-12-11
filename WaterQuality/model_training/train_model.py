# WaterQuality/model_training/train_model.py
from sklearn.model_selection import train_test_split
import time

def train_and_evaluate_model(X, y, model, hyperparams):
    """
    Обучает модель, используя параметры из конфигурационного файла
    """
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=hyperparams['train_test_split']['test_size'],
        random_state=hyperparams['train_test_split']['random_state']
    )

    # Замер времени обучения
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=hyperparams['epochs'],
        batch_size=hyperparams['batch_size'],
        validation_split=hyperparams['validation_split'],
        verbose=2
    )
    training_time = time.time() - start_time

    # Получение всех метрик из истории обучения
    metrics = {}
    for metric_name, values in history.history.items():
        metrics[metric_name] = values[-1]
        metrics[f'best_{metric_name}'] = max(values) if 'loss' not in metric_name else min(values)

    metrics['epochs_trained'] = len(history.history['loss'])
    metrics['training_time'] = training_time

    # Получение предсказаний
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    return {
        'history': history.history,
        'metrics': metrics
    }