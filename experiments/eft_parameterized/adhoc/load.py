import os
import re

import numpy as np
import pandas as pd

from experiments.eft_parameterized.adhoc.data_reader import DataTable


def parse_filename(filename):
    regex = re.compile(r'm?\dp\d')
    theta, = regex.findall(filename)
    theta = theta.replace('m', '-')
    theta = theta.replace('p', '.')
    theta = float(theta)
    return theta


def load_data(filepath):
    tab = DataTable(fname=filepath)
    data = pd.DataFrame()
    for i, (observable_name, observable_type) in enumerate(tab.get_observables_and_types()):
        if observable_type == float:
            data[observable_name] = tab.data[:, i]
    return data


def main():
    data_dir = 'data'
    out_dir = '../data'

    for fname in os.listdir(data_dir):
        theta = parse_filename(fname)
        fpath = os.path.join(data_dir, fname)
        data = load_data(fpath)
        filtered_data = data[~(data == -99).all(axis=1)].values
        data.to_csv(os.path.join(out_dir, f'{theta}.csv'))
        np.save(os.path.join(out_dir, f'{theta}.npy'), filtered_data)


if __name__ == '__main__':
    main()
