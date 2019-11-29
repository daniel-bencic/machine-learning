import numpy as np

def standardize_data(data):
    return data.apply(lambda x: (x - data[x.name].mean()) / data[x.name].std()).to_numpy()

def calc_pca(data):
    # standardize variables
    m = standardize_data(data)
    return np.linalg.svd(m, full_matrices=False)
