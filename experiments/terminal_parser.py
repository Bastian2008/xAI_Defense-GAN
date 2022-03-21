import argparse

parser = argparse.ArgumentParser(description='Parse models and attacks')
parser.add_argument('--models', metavar='model', type=str, nargs='+',
                    help='List of models that are going to be evaluated. By default all models are evaluated. \
                    The accepted values are: model_a, model_b, model_c, model_d, model_e, model_f, model_g')
parser.add_argument('--dss', metavar='ds', type=str, nargs='+', 
                    help='List of datasets that the models will be tested against. By default all datasets are used. \
                        The allowed values are: mnist, cw, fgsm, pgd')


