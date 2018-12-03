import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
def load_balanced():
    data = pd.read_table('balance-scale.data.txt',sep=',',header=None)
    X_raw, y_raw = data.ix[:,1:],data.ix[:,0]
    X = scale(X_raw,axis=1)
    y = np.empty_like(y_raw,dtype='int64')
    attrs = pd.unique(y_raw)
    k = 0
    for attr in attrs:
        y[y_raw == attr] = k
        k+=1
    return X,y

if __name__ == '__main__':
    X,y = load_balanced()
    print(X)