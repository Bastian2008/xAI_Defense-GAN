from pathlib import Path
import os
import re
import pickle
import numpy as np
import argparse

def join_reconstruction(path, all_files):
  rec_list = [None] * len(all_files)
  for name in (all_files):
    position = int(re.sub('^[^_]*_|_[^_]*$', '', name))
    with open(f'{path}{name}', 'rb') as file:
      rec_list[position] = pickle.load(file, encoding='latin1')
  return rec_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse models and attacks')
    parser.add_argument('input', type=str, help='Path to the directory where all reconstructions files can be found')
    parser.add_argument('out', type=str, help='Path to the directory where the union of all files will be stored')
    args = parser.parse_args()

    test_files = [f for f in os.listdir(args.input) if '.pkl' in f]
    test_list = join_reconstruction(args.input, test_files)

    np_rec = np.concatenate(test_list).reshape(len(test_files), 28, 28)
    with open(f'{args.out}test_rec.pkl', 'wb') as f:
        pickle.dump(np_rec, f)