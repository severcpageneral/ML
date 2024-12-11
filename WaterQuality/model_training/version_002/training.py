# Базовые библиотеки для работы с данными и вычислений
import pandas as pd

# Системные библиотеки
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 - INFO, 1 - WARNING, 2 - ERROR, 3 - FATAL
import yaml
import time

# Модули создания, обучения модели и логирования
sys.path.append(os.path.join(os.getcwd(), ".."))
from create_model import create_model
from train_model import train_and_evaluate_model


# Загружаем датасет
sys.path.append(os.path.join(os.getcwd(), ".."))
file_path = os.path.join("..", "..", "data", "raw", "water_potability.csv")

df = pd.read_csv(file_path)
X = df.drop('Potability', axis=1)
y = df['Potability']
feature_names = X.columns.tolist()

## Загружаем конфиги
def load_config(file_name):
    config_path = os.path.join(os.getcwd(), 'configs', file_name)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

hyperparams = load_config('hyperparameters.yml')['hyperparameters']
model_config = load_config('model_config.yml')['model']
logging_config = load_config('logging_config.yml')['logging']

# Создание модели
model = create_model(input_dim=X.shape[1], model_config=model_config)

# Обучение и оценка модели
results = train_and_evaluate_model(
    X=X,
    y=y,
    model=model,
    hyperparams=hyperparams
)

# После получения results добавляем:
run_dir = os.path.join("..", "..", "run")
os.makedirs(run_dir, exist_ok=True)

history_file = os.path.join(run_dir, "run_version_002_history.csv")
metrics_file = os.path.join(run_dir, "run_version_002_metrics.csv")

# 1. Сохраняем history
history_df = pd.DataFrame(results['history'])
history_df['run_version'] = f"run_version_002_{int(time.time())}"
# Нумеруем новые строки с 1
history_df['epoch'] = range(1, len(history_df) + 1)

# Если файл существует - дочитываем старые данные и добавляем новые
if os.path.exists(history_file):
    existing_history = pd.read_csv(history_file)
    history_df = pd.concat([existing_history, history_df], ignore_index=True)

history_df.to_csv(history_file, index=False)

# 2. Сохраняем metrics
metrics_df = pd.DataFrame([results['metrics']])
metrics_df['run_version'] = f"run_version_002_{int(time.time())}"

# Если файл существует - дочитываем старые данные и добавляем новые
if os.path.exists(metrics_file):
    existing_metrics = pd.read_csv(metrics_file)
    metrics_df = pd.concat([existing_metrics, metrics_df], ignore_index=True)

metrics_df.to_csv(metrics_file, index=False)

# 3. Сохраняем модель
models_dir = os.path.join("..", "..", "models")
os.makedirs(models_dir, exist_ok=True)
model.save(os.path.join(models_dir, f"run_version_002_{int(time.time())}.keras"))

print(f"Results saved in directory: {run_dir}")
print(f"Files saved:")
print(f"- run_version_002_history.csv")
print(f"- run_version_002_metrics.csv")
print(f"Model saved as: run_version_002_{int(time.time())}.keras")