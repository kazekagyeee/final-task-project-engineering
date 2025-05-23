import os
import numpy as np
import pandas as pd

# Создаем папки train и test, если их нет
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

# Функция для генерации данных
def generate_data(num_samples, add_noise=False, add_anomalies=False):
    # Генерация данных: y = 2x + 10 + noise + anomalies
    x = np.linspace(0, 10, num_samples)
    noise = np.random.normal(0, 0.5, num_samples) if add_noise else 0
    y = 2 * x + 10 + noise

    if add_anomalies:
        # Добавляем аномалии в случайные точки
        anomaly_indices = np.random.choice(num_samples, size=int(num_samples * 0.1), replace=False)
        y[anomaly_indices] += np.random.normal(0, 10, len(anomaly_indices))

    return pd.DataFrame({'x': x, 'y': y})

# Генерация тренировочных данных
regular_data = generate_data(100)  # Обычные данные
noised_data = generate_data(100, add_noise=True)  # Данные с шумом
data_with_anomalies = generate_data(100, add_anomalies=True)  # Данные с аномалиями

# Сохранение тренировочных данных в папку "train"
regular_data.to_csv('train/train_data_1.csv', index=False)
noised_data.to_csv('train/train_data_2.csv', index=False)
data_with_anomalies.to_csv('train/train_data_3.csv', index=False)

# Генерация тестовых данных
test_regular_data = generate_data(20)  # Обычные данные
test_noised_data = generate_data(20, add_noise=True)  # Данные с шумом
test_data_with_anomalies = generate_data(20, add_anomalies=True)  # Данные с аномалиями

# Сохранение тестовых данных в папку "test"
test_regular_data.to_csv('test/test_data_1.csv', index=False)
test_noised_data.to_csv('test/test_data_2.csv', index=False)
test_data_with_anomalies.to_csv('test/test_data_3.csv', index=False)

print("Данные успешно созданы и сохранены в папках 'train' и 'test'.")