import os
import csv
import glob
import numpy as np
import collections


Prices = collections.namedtuple('Prices', field_names=['open', 'high', 'low', 'close'])


def read_csv(file_name, sep=','):
    print("Reading", file_name)
    with open(file_name, 'rt', encoding='utf-8') as fd:
        reader = csv.reader(fd, delimiter=sep)
        h = next(reader)
        if '<OPEN>' not in h and sep == ',':
            return read_csv(file_name, ';')
        o_idx, c_idx, h_idx, l_idx = [h.index(s) for s in ('<OPEN>', '<CLOSE>', '<HIGH>', '<LOW>')]
        o, c, h, l = [], [], [], []
        for row in reader:
            o.append(float(row[o_idx]))
            c.append(float(row[c_idx]))
            h.append(float(row[h_idx]))
            l.append(float(row[l_idx]))
    return Prices(open=np.array(o, dtype=np.float32),
                  high=np.array(h, dtype=np.float32),
                  low=np.array(l, dtype=np.float32),
                  close=np.array(c, dtype=np.float32))


def prices_to_relative(prices):
    """
    Convert prices to relative in respect to open price
    :param ochl: tuple with open, close, high, low 
    :return: tuple with open, rel_close, rel_high, rel_low 
    """
    assert isinstance(prices, Prices)
    rh = (prices.high - prices.open) / prices.open
    rl = (prices.low - prices.open) / prices.open
    rc = (prices.close - prices.open) / prices.open
    return Prices(open=prices.open, high=rh, low=rl, close=rc)


def load_relative(csv_file):
    return prices_to_relative(read_csv(csv_file))


def price_files(dir_name):
    result = []
    for path in glob.glob(os.path.join(dir_name, "*.csv")):
        name = os.path.basename(path).split("_", maxsplit=1)[0]
        result.append((name, path))
    return result
