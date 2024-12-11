import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import (
    Precision, Recall, AUC, F1Score,
    FalsePositives, FalseNegatives,
    TruePositives, TrueNegatives,
    PrecisionAtRecall, RecallAtPrecision,
    SensitivityAtSpecificity, SpecificityAtSensitivity,
    FBetaScore
)

def create_model(input_dim, model_config):
    """
    Создание архитектуры нейронной сети на основе конфигурации
    """
    # Создаем модель, используя функциональный API
    inputs = keras.Input(shape=(input_dim,))
    x = inputs

    # Создаем слои согласно конфигурации
    for layer in model_config['layers']:
        x = layers.Dense(layer['units'], activation=layer['activation'])(x)
        if 'dropout' in layer:
            x = layers.Dropout(layer['dropout'])(x)

    outputs = x

    # Создаем модель
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_config['name'])

    # Подготавливаем метрики
    metrics = []
    for metric in model_config['compile']['metrics']:
        if isinstance(metric, str):
            metrics.append(metric)
        else:
            metric_class = globals()[metric['class']]
            metrics.append(metric_class(**metric['config']))

    # Компиляция модели
    model.compile(
        optimizer=model_config['compile']['optimizer'],
        loss=model_config['compile']['loss'],
        metrics=metrics
    )

    return model