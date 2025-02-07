import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Завантаження збереженої моделі
model = load_model('digit_recognizer.h5')

# Завантаження зображення для розпізнавання (переконайтесь, що воно знаходиться в поточній директорії або вкажіть повний шлях)
# Файл повинен містити зображення цифри, бажано у відтінках сірого (MNIST має формат 28x28 пікселів)
img = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Не вдалося завантажити зображення. Перевірте шлях до файлу.")
else:
    # Якщо розмір зображення не 28x28, змінюємо його розмір
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))

    # Нормалізація пікселів до діапазону [0, 1]
    img = img.astype('float32') / 255.0

    # Зміна форми зображення для подачі в модель: (1, 28, 28, 1)
    img = np.expand_dims(img, axis=0)  # додаємо розмір для batch
    img = np.expand_dims(img, axis=-1)  # додаємо вимір каналу

    # Отримання прогнозу від моделі
    predictions = model.predict(img)
    predicted_digit = np.argmax(predictions)

    print("Передбачувана цифра:", predicted_digit)
