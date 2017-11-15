import os
import csv
import glob
import numpy as np
import collections


Prices = collections.namedtuple('Prices', field_names=['open', 'high', 'low', 'close'])


def read_csv(file_name, sep=',', filter_data=False):
    print("Reading", file_name)
    with open(file_name, 'rt', encoding='utf-8') as fd:
        reader = csv.reader(fd, delimiter=sep)
        h = next(reader)
        if '<OPEN>' not in h and sep == ',':
            return read_csv(file_name, ';')
        indices = [h.index(s) for s in ('<OPEN>', '<CLOSE>', '<HIGH>', '<LOW>')]
        o, c, h, l = [], [], [], []
        count_out = 0
        count_filter = 0
        for row in reader:
            vals = list(map(float, [row[idx] for idx in indices]))
            if filter_data and all(map(lambda v: abs(v-vals[0]) < 1e-3, vals)):
                count_filter += 1
                continue

            po, pc, ph, pl = vals
            count_out += 1
            o.append(po)
            c.append(pc)
            h.append(ph)
            l.append(pl)
    print("Read done, %d rows filtered from %d total" % (count_filter, count_filter + count_out))
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
        result.append(path)
    return result


def load_year_data(year, basedir='data'):
    y = str(year)[-2:]
    result = {}
    for path in glob.glob(os.path.join(basedir, "*_%s*.csv" % y)):
        result[path] = load_relative(path)
    return result
