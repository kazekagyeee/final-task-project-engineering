import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Функция для загрузки данных из папки
def load_data_from_folder(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Папка '{folder_path}' не найдена.")

    data_frames = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            try:
                df = pd.read_csv(file_path)
                data_frames.append(df)
            except Exception as e:
                print(f"Ошибка при чтении файла '{file_name}': {e}")

    if not data_frames:
        raise ValueError(f"Нет CSV файлов в папке '{folder_path}'.")

    return pd.concat(data_frames, ignore_index=True)


# Функция для предобработки данных
def preprocess_data(data, scaler=None):
    # Инициализация StandardScaler
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)

    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    return scaled_df, scaler


# Путь к папкам train и test
train_folder = 'train'
test_folder = 'test'

# Загрузка данных из папок
train_data = load_data_from_folder(train_folder)
test_data = load_data_from_folder(test_folder)

# Предобработка тренировочных данных
scaled_train_data, train_scaler = preprocess_data(train_data)

# Предобработка тестовых данных
# Важно: используем параметры scaler, обученные на тренировочных данных
scaled_test_data = pd.DataFrame(train_scaler.transform(test_data), columns=test_data.columns)

# Сохранение предобработанных данных
scaled_train_data.to_csv('train/scaled_train_data.csv', index=False)
scaled_test_data.to_csv('test/scaled_test_data.csv', index=False)

print("Данные успешно предобработаны и сохранены.")