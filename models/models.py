from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


model_a = keras.Sequential([
    keras.layers.Conv2D(64, 5, strides=1, activation='relu', padding='same', input_shape=(28,28,1)),
    keras.layers.Conv2D(64, 5, strides=2, activation='relu', padding='valid'),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

model_b = keras.Sequential([
    keras.layers.Dropout(0.2, input_shape=(28,28,1)),
    keras.layers.Conv2D(64, 8, strides=2, activation='relu', padding='same'),
    keras.layers.Conv2D(128, 6, strides=2, activation='relu', padding='valid'),
    keras.layers.Conv2D(128, 5, strides=1, activation='relu', padding='valid'),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

model_c = keras.Sequential([
    keras.layers.Conv2D(128, 3, strides=1, activation='relu', padding='same', input_shape=(28,28,1)),
    keras.layers.Conv2D(64, 5, strides=2, activation='relu', padding='valid'),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

model_d = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

model_e = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model_f = keras.Sequential([
    keras.layers.Conv2D(64, 8, strides=2, activation='relu', padding='same', input_shape=(28,28,1)),
    keras.layers.Conv2D(128, 6, strides=2, activation='relu', padding='valid'),
    keras.layers.Conv2D(128, 5, strides=1, activation='relu', padding='valid'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

model_g = keras.Sequential([
    keras.layers.Conv2D(6, 5, activation='relu', input_shape=(28,28,1)),
    keras.layers.Conv2D(12, 5, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.3),
    keras.layers.Conv2D(24, 5, activation='relu'),
    keras.layers.Conv2D(48,5, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Flatten(),
    keras.layers.Dense(120, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(84, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation='softmax')
])

if __name__ == '__main__':

    models = {'Model A': model_a, 'Model B': model_b,'Model C': model_c, 'Model D': model_d, 'Model E': model_e, 'Model F': model_f, 'Model G' : model_g}
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    for name, model in models.items():
        print(f'training {name}')
        model.fit(x_train, y_train, epochs=200, batch_size=32)
        model.evaluate(x_test, y_test)
        name = name.lower().replace(' ', '_')
        model.save(f'./{name}')