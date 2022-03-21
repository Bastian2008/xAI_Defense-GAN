import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from terminal_parser import parser
import pickle
import utils
import os


if __name__ == '__main__':
    models = [('model_a', 'conv2d_1'), ('model_b', 'conv2d_3'), ('model_c', 'conv2d_6'), ('model_f', 'conv2d_8'), ('model_g', 'conv2d_3')]
    last_conv_mapping = {model:layer for model,layer in models}
    args = parser.parse_args()
    adv_ds = args.dss if args.dss else ['cw', 'fgsm', 'pgd', 'mnist']
    models = [(model, last_conv_mapping[model]) for model in args.models] if args.models else models
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32')/255
    y_test = to_categorical(y_test)
    dirname = os.path.dirname(__file__)

    for ds in adv_ds:
        with open(f'{dirname}/../datasets/{ds}/test_rec.pkl', 'rb') as f:
            rec_test = pickle.load(f)
        if ds != 'mnist':  
            fd = open(f'{dirname}/../datasets/{ds}/t10k-images-idx3-ubyte', 'r')
            loaded = np.fromfile(file=fd, dtype=np.uint8)
            adv_test = loaded[16:].reshape((10000, 28, 28)).astype(np.float)/255
        else:
            adv_test = x_test
        print('\n', ds.upper())
        for model, last_conv in models:
            tf_model = tf.keras.models.load_model(f'{dirname}/../models/{model}')
            tf_model.layers[-1].activation = None
            xai_test = utils.gradcam_ds(adv_test, rec_test, tf_model, last_conv)
            tf_model.layers[-1].activation = tf.keras.activations.softmax
            print( model, ': ', tf_model.evaluate(xai_test, y_test))