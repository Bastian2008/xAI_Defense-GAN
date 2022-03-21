import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from terminal_parser import parser
import pickle
import os


if __name__ == '__main__':

    args = parser.parse_args()
    adv_ds = args.dss if args.dss else ['cw', 'fgsm', 'pgd', 'mnist']
    models = args.models if args.models else ['model_a', 'model_b', 'model_c', 'model_d', 'model_e', 'model_f', 'model_g']
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32')/255
    y_test = to_categorical(y_test)
    dirname = os.path.dirname(__file__)

    for ds in adv_ds:
        with open(f'{dirname}/../datasets/{ds}/test_rec.pkl', 'rb') as f:
            loaded = pickle.load(f)
        print('\n', ds.upper())
        for model in models:
            tf_model = tf.keras.models.load_model(f'{dirname}/../models/{model}')
            print( model, ': ', tf_model.evaluate(loaded, y_test))