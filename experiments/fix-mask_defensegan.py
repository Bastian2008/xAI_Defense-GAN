import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from terminal_parser import parser
import pickle
import utils
import os


if __name__ == '__main__':
    parser.add_argument('--mask', type=str, nargs='?', 
                    help='Fixed mask that will be used. By default random mask is used. Allowed values: random, center, margin')

    args = parser.parse_args()
    adv_ds = args.dss if args.dss else ['cw', 'fgsm', 'pgd', 'mnist']
    models = args.models if args.models else ['model_a', 'model_b', 'model_c', 'model_d', 'model_e', 'model_f', 'model_g']
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32')/255
    y_test = to_categorical(y_test)
    mask_function = {'random': utils.random_mask, 'center': utils.center_mask, 'margin': utils.margin_mask}.get(args.mask, utils.center_mask)
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
        xai_test = mask_function(adv_test, rec_test)
        print('\n', ds.upper())
        for model in models:
            tf_model = tf.keras.models.load_model(f'{dirname}/../models/{model}')
            print( model, ': ', tf_model.evaluate(xai_test, y_test)) 
    
