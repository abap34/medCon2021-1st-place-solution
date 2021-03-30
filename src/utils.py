import os

import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm


def read_data(data_contain_path):
    train = pd.read_csv(data_contain_path + '/train.csv')
    test = pd.read_csv(data_contain_path + '/test.csv')
    train["is_train"] = True
    test["is_train"] = False
    submit = pd.read_csv(data_contain_path + '/sample_submission.csv')

    return train, test, submit


def read_wave(paths):
    X = []
    for p in tqdm.tqdm(paths):
        X.append(np.load(p))
    return np.stack(X)


def seed_everything(seed=0):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
