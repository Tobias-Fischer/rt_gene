import os
from argparse import ArgumentParser

import h5py

root_dir = os.path.dirname(os.path.realpath(__file__))

root_parser = ArgumentParser(add_help=False)
root_parser.add_argument('--hdf5_file', type=str, required=True, help='path to the datasets to combine', action="append")
root_parser.add_argument('--save_file', type=str, default=os.path.abspath(os.path.join(root_dir, "combined_dataset.hdf5")))
params = root_parser.parse_args()

assert len(params.hdf5_file) >= 2, "Need at least two datasets to combine"

files = [h5py.File(path, mode="r") for path in params.hdf5_file]
keys = [list(h5.keys()) for h5 in files]
_ = [h5.close() for h5 in files]

h5_all = h5py.File(params.save_file, mode='w')

s_idx = 0
for dataset, path in zip(keys, params.hdf5_file):
    for key in dataset:
        h5_all[str("s{:03d}".format(s_idx))] = h5py.ExternalLink(path, key)
        s_idx += 1

h5_all.flush()
h5_all.close()
