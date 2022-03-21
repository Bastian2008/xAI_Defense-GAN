import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from terminal_parser import parser
import pickle


if __name__ == '__main__':

    args = parser.parse_args()
    adv_ds = args.dss if args.dss else ['cw', 'fgsm', 'pgd', 'mnist']
    models = args.model if args.model else ['model_a', 'model_b', 'model_c', 'model_d', 'model_e', 'model_f', 'model_g']
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32')/255
    y_test = to_categorical(y_test)

    for ds in adv_ds:
        with open(f'../datesets/{ds}/test_rec.pkl', 'rb') as f:
            loaded = pickle.load(f)
        print('\n', ds, ': ', loaded.shape)
        for model in models:
            tf_model = tf.keras.models.load_model(f'../models/{model}')
            print( model, ': ', tf_model.evaluate(loaded, y_test))