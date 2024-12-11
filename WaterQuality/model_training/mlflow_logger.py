# WaterQuality/model_training/mlflow_logger.py
import os
import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException


def log_to_mlflow(evaluation_results, experiment_name, logging_config, run_name, save_model, hyperparams=None,
                  model_config=None):
    """
    Логирует результаты эксперимента в MLflow

    Parameters:
    -----------
    evaluation_results : dict
        Результаты обучения и оценки модели
    experiment_name : str
        Название эксперимента в MLflow
    logging_config : dict
        Конфигурация логирования
    run_name : str
        Название текущего запуска
    save_model : bool
        Флаг для сохранения модели
    hyperparams : dict, optional
        Гиперпараметры модели
    model_config : dict, optional
        Конфигурация архитектуры модели
    """
    try:
        experiment_id = mlflow.create_experiment(logging_config['experiment_name'])
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(logging_config['experiment_name']).experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        # Логирование тегов эксперимента
        for tag, value in logging_config['tags'].items():
            mlflow.set_tag(tag, value)

        # Извлечение данных из результатов
        model = evaluation_results['model']
        history = evaluation_results['history']
        metrics = evaluation_results['metrics']
        X_test_scaled = evaluation_results['data']['X_test_scaled']
        y_pred = evaluation_results['predictions']['y_pred']

        # Логирование параметров модели
        model_params = {
            "input_dim": evaluation_results['data']['X_test_scaled'].shape[1],
            "optimizer": model.optimizer.get_config()['name'],
            "loss": model.loss,
            "metrics": model.metrics_names,
        }

        # Логирование гиперпараметров, если они предоставлены
        if hyperparams:
            # Преобразуем вложенные словари в плоский формат
            flat_hyperparams = {}
            for key, value in hyperparams.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_hyperparams[f"{key}_{sub_key}"] = sub_value
                else:
                    flat_hyperparams[key] = value

            model_params.update({f"hyper_{k}": v for k, v in flat_hyperparams.items()})

        # Логирование конфигурации архитектуры модели, если она предоставлена
        if model_config:
            # Добавляем основные параметры архитектуры
            arch_params = {
                "architecture_name": model_config.get('name', 'unknown'),
                "n_layers": len(model_config.get('layers', [])),
            }
            # Добавляем информацию о каждом слое
            for i, layer in enumerate(model_config.get('layers', [])):
                arch_params[f"layer_{i}_units"] = layer.get('units')
                arch_params[f"layer_{i}_activation"] = layer.get('activation')
                if 'dropout' in layer:
                    arch_params[f"layer_{i}_dropout"] = layer.get('dropout')

            model_params.update({f"arch_{k}": v for k, v in arch_params.items()})

        # Логируем все параметры
        mlflow.log_params(model_params)

        # Логирование финальных значений метрик
        for metric_name, metric_value in metrics.items():
            if metric_name != 'training_time' and metric_name != 'epochs_trained':
                mlflow.log_metric(f"final_{metric_name}", metric_value)

        # Логирование метрик по эпохам
        for metric_name, values in history.items():
            for epoch, value in enumerate(values):
                mlflow.log_metric(metric_name, value, step=epoch)

        # Логирование специальных метрик
        mlflow.log_metric("training_time", metrics['training_time'])
        mlflow.log_metric("epochs_trained", metrics['epochs_trained'])

        # Логирование истории обучения
        for metric_name, values in history.items():
            for epoch, val in enumerate(values):
                mlflow.log_metric(metric_name, val, step=epoch)

        # Сохранение модели и артефактов
        if save_model:
            # Сохраняем модель с помощью MLflow
            signature = infer_signature(X_test_scaled, y_pred)
            mlflow.tensorflow.log_model(
                model,
                run_name,
                signature=signature,
                registered_model_name=run_name
            )
            # Сохранение в локальной директории models
            local_model_path = os.path.join("..", "..", "models", f"{run_name}.keras")
            model.save(local_model_path)