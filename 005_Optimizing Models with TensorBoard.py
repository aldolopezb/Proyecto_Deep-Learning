import tensorflow as tf  # Importa TensorFlow
from tensorflow.keras.datasets import cifar10  # Importa el conjunto de datos CIFAR-10
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Importa el generador de imágenes de Keras
from tensorflow.keras.models import Sequential  # Importa el modelo secuencial de Keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten  # Importa las capas Dense, Dropout, Activation y Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D  # Importa las capas Conv2D y MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard  # Importa el callback TensorBoard para la visualización del entrenamiento

import pickle  # Importa la biblioteca pickle para la serialización de datos
import numpy as np  # Importa NumPy para operaciones numéricas
import time  # Importa la biblioteca time para trabajar con funciones relacionadas con el tiempo

# Carga los datos desde los archivos pickle
# Abre el archivo "X.pickle" en modo lectura binaria y carga los datos de las imágenes como un array de NumPy
pickle_in = open("X.pickle", "rb")
X = np.array(pickle.load(pickle_in))

# Abre el archivo "y.pickle" en modo lectura binaria y carga las etiquetas de las imágenes como un array de NumPy
pickle_in = open("y.pickle", "rb")
y = np.array(pickle.load(pickle_in))

# Normaliza los datos
# Divide cada valor de píxel por 255.0 para normalizar los datos de las imágenes a un rango [0, 1]
X = X / 255.0

# Define los parámetros para la búsqueda de hiperparámetros
dense_layers = [1]  # Lista con el número de capas densas a probar
layer_sizes = [32]  # Lista con los tamaños de capa a probar
conv_layers = [3]  # Lista con el número de capas convolucionales a probar

# Realiza la búsqueda de hiperparámetros
for dense_layer in dense_layers:  # Itera sobre el número de capas densas
    for layer_size in layer_sizes:  # Itera sobre los tamaños de capa
        for conv_layer in conv_layers:  # Itera sobre el número de capas convolucionales
            # Define un nombre único para el experimento de TensorBoard
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir="lot_logs/{}".format(NAME))  # Inicializa el callback de TensorBoard con el directorio de logs especificado
            print(NAME)  # Imprime el nombre del experimento

            model = Sequential()  # Inicializa un modelo secuencial

            # Añade la primera capa convolucional con el tamaño de capa especificado
            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))  # Añade una función de activación ReLU
            model.add(MaxPooling2D(pool_size=(2, 2)))  # Añade una capa de max-pooling con un tamaño de pool de 2x2

            # Añade capas convolucionales adicionales según el número especificado
            for l in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))  # Añade una función de activación ReLU
                model.add(MaxPooling2D(pool_size=(2, 2)))  # Añade una capa de max-pooling con un tamaño de pool de 2x2

            model.add(Flatten())  # Aplana las características 3D a un vector 1D

            # Añade capas densas según el número especificado
            for _ in range(dense_layer):
                model.add(Dense(512))  # Añade una capa densa con 512 unidades
                model.add(Activation('relu'))  # Añade una función de activación ReLU

            model.add(Dense(1))  # Añade una capa densa con una unidad (salida)
            model.add(Activation('sigmoid'))  # Añade una función de activación sigmoide para la salida

            # Compila el modelo
            # Utiliza la pérdida de entropía cruzada binaria y el optimizador Adam
            # Mide la precisión durante el entrenamiento
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Entrena el modelo
            # Utiliza un tamaño de lote de 32 y entrena por 7 épocas
            # Usa el 30% de los datos para la validación durante el entrenamiento
            model.fit(X, y, batch_size=32, epochs=7, validation_split=0.3, callbacks=[tensorboard])

#Ejecutamos el siguiente comando tensorboard --logdir=lot_logs/
#en la ruta donde se tiene guardado el codigo
#Despues de abrir nuestro localhost, se va a scalars y escribimos \w