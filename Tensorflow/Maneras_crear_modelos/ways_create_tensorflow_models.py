''' MANERAS DE CREAR MODELOS EN TENSORFLOW '''

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict


''' CARGANDO LOS DATOS '''
datos_entrenamiento, datos_prueba = tf.keras.datasets.fashion_mnist.load_data()
imagenes_entrenamiento, clases_entrenamiento = datos_entrenamiento
imagenes_prueba, clases_prueba = datos_prueba


class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num_classes = len(class_names)


''' ALGUNAS IMAGENES '''

plt.figure(figsize=(10, 10))
for i in range(36):
    plt.subplot(6, 6, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imagenes_entrenamiento[i].reshape((28, 28)),
               cmap='gray',
               vmin=0,
               vmax=255)
    label_index = int(clases_entrenamiento[i])
    plt.title(class_names[label_index])
plt.show()


''' PREPARANDO LOS DATOS '''
imagenes_entrenamiento = imagenes_entrenamiento / 255.0
imagenes_prueba = imagenes_prueba / 255.0

input_shape = [28, 28, 1]
imagenes_entrenamiento = imagenes_entrenamiento.reshape([imagenes_entrenamiento.shape[0]] + input_shape)
imagenes_prueba = imagenes_prueba.reshape([imagenes_prueba.shape[0]] + input_shape)


''' FUNCION DE ENTRENAMIENTO '''
def entrena_modelo(modelo: tf.keras.models.Model) -> Dict:
    modelo.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
    history = modelo.fit(
        imagenes_entrenamiento, 
        clases_entrenamiento, 
        epochs=20,
        batch_size=1024,
        validation_data=(imagenes_prueba, clases_prueba)
        )

    print(modelo.summary())

    return history.history


''' SECUENCIAL '''
def retorna_modelo_secuencial() -> tf.keras.models.Model:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(3, 3),
                                     activation='relu',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.25))
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.25))
    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(3, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.4))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128,
                                    activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(units=num_classes,
                                    activation='softmax'))
    return model

modelo_secuencial = retorna_modelo_secuencial()
historia_secuencial = entrena_modelo(modelo_secuencial)


def retorna_modelo_secuencial_lista() -> tf.keras.models.Model:
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(rate=0.25),
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(rate=0.25),
            tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=(3, 3),
                                   activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(rate=0.4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128,
                                  activation='relu'),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Dense(units=num_classes,
                                  activation='softmax')
        ]
    )
    return model


modelo_secuencial_lista = retorna_modelo_secuencial_lista()
historia_secuencial_lista = entrena_modelo(modelo_secuencial_lista)


''' FUNCIONAL '''
def retorna_modelo_funcional() -> tf.keras.models.Model:
    entrada = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=(3, 3),
                               activation='relu')(entrada)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(rate=0.25)(x)
    x = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(rate=0.25)(x)
    x = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=(3, 3),
                               activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(rate=0.4)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=128,
                              activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    salida = tf.keras.layers.Dense(units=num_classes,
                                   activation='softmax')(x)
    model = tf.keras.models.Model(inputs=entrada,
                                  outputs=salida)
    return model


modelo_funcional = retorna_modelo_funcional()
historia_funcional = entrena_modelo(modelo_funcional)


''' CLASES '''
class ModeloClase(tf.keras.models.Model):
    def __init__(self) -> None:
        super().__init__()
        self.cnn_1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(3, 3),
                                            activation='relu',
                                            input_shape=input_shape)
        self.pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout_1 = tf.keras.layers.Dropout(rate=0.25)
        self.cnn_2 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(3, 3),
                                            activation='relu')
        self.pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout_2 = tf.keras.layers.Dropout(rate=0.25)
        self.cnn_3 = tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=(3, 3),
                                            activation='relu')
        self.pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout_3 = tf.keras.layers.Dropout(rate=0.4)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=128,
                                           activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=0.3)
        self.salida = tf.keras.layers.Dense(units=num_classes,
                                            activation='softmax')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        x = self.cnn_1(x)
        x = self.pool_1(x)
        x = self.dropout_1(x)
        x = self.cnn_2(x)
        x = self.pool_2(x)
        x = self.dropout_2(x)
        x = self.cnn_3(x)
        x = self.pool_3(x)
        x = self.dropout_3(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.salida(x)
        return x


modelo_clase = ModeloClase()
historia_clase = entrena_modelo(modelo_clase)


'''
VISUALIZANDO
'''

val_loss_df = pd.DataFrame({
    'secuencial': historia_secuencial['val_loss'],
    'secuencial_lista': historia_secuencial_lista['val_loss'],
    'funcional': historia_funcional['val_loss'],
    'clase': historia_clase['val_loss']
})


val_loss_df.plot()



''' CLASES + LAYERS '''
class CNNBloque(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: Tuple[int, int],
                 activation: str, pool_size: Tuple[int, int],
                 rate: float) -> None:
        super().__init__()
        self.cnn = tf.keras.layers.Conv2D(filters=filters,
                                          kernel_size=kernel_size,
                                          activation=activation)
        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size)
        self.dropout = tf.keras.layers.Dropout(rate=rate)
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.cnn(inputs)
        x = self.max_pool(x)
        x = self.dropout(x)
        return x


class ModeloClasesLayer(tf.keras.models.Model):
    def __init__(self) -> None:
        super().__init__()
        self.cnn_bloque_lista = [
            CNNBloque(filters=32, kernel_size=(3, 3),
                      activation='relu', pool_size=(2, 2),
                      rate=0.25),
            CNNBloque(filters=64, kernel_size=(3, 3),
                      activation='relu', pool_size=(2, 2),
                      rate=0.25),
            CNNBloque(filters=128, kernel_size=(3, 3),
                      activation='relu', pool_size=(2, 2),
                      rate=0.4)
        ]
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=128,
                                           activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=0.3)
        self.softmax = tf.keras.layers.Dense(units=num_classes,
                                             activation='softmax')
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        for layer in self.cnn_bloque_lista:
            x = layer(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.softmax(x)
        return x


modelo_clase_layer = ModeloClasesLayer()
historia_clase_layer = entrena_modelo(modelo_clase_layer)


