import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Завантаження датасету MNIST (цифри від 0 до 9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Попередня обробка даних:
# - Зміна розміру зображень (28x28) для згорткової мережі (додавання виміру каналу)
# - Нормалізація пікселів (значення від 0 до 1)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Перетворення міток (labels) у формат one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Створення моделі згорткової нейронної мережі
model = Sequential([
    # Перший згортковий шар з 32 фільтрами, розмір ядра 3x3 та функцією активації ReLU
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Шар підвибірки (max pooling) для зменшення розмірності
    MaxPooling2D(pool_size=(2, 2)),

    # Другий згортковий шар з 64 фільтрами
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    # Ще один шар підвибірки
    MaxPooling2D(pool_size=(2, 2)),

    # Перетворення багатовимірного вектора в одномірний
    Flatten(),
    # Щільний шар з 128 нейронами
    Dense(128, activation='relu'),
    # Dropout для запобігання перенавчанню (50% випадкове відключення нейронів)
    Dropout(0.5),
    # Вихідний шар з 10 нейронами (по одному для кожної цифри) та softmax активацією
    Dense(10, activation='softmax')
])

# Компіляція моделі з оптимізатором 'adam' та функцією втрат categorical_crossentropy
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Навчання моделі:
# batch_size - розмір пакету, epochs - кількість епох навчання
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# Оцінка моделі на тестовому наборі
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

# Збереження моделі для подальшого використання
model.save("digit_recognizer.h5")
