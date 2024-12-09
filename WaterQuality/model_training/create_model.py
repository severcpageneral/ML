# WaterQuality/model_training/create_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_dim, model_config):
    """
    Создание архитектуры нейронной сети на основе конфигурации
    """
    # Создаем модель, используя функциональный API
    inputs = keras.Input(shape=(input_dim,))
    x = inputs

    for layer in model_config['layers']:
        x = layers.Dense(layer['units'], activation=layer['activation'])(x)
        if 'dropout' in layer:
            x = layers.Dropout(layer['dropout'])(x)

    outputs = x

    # Создаем модель
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_config['name'])

    # Компиляция модели
    model.compile(
        optimizer=model_config['compile']['optimizer'],
        loss=model_config['compile']['loss'],
        metrics=model_config['compile']['metrics']
    )

    return model