import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy.linalg import inv
import matplotlib.pyplot as plt # for data visualization
import matplotlib
matplotlib.use('Qt5Agg')  # or another suitable backend for your environment


def calc_Em(X):
    # using PCA to dimensionality reduction
    # covariance matrix of the data is [NxN]
    # Em = use ONLY the first M columns of the Covariance Matrix [NxM]
    N = np.shape(X)[1]
    M = 2
    pca = PCA(n_components=M)
    Y = pca.fit_transform(X) # fit the model to the input data
    pca_vec = pca.components_ # PCA eigen vectors [MxN]
    print(f"PCA eigen vals:{pca.explained_variance_ratio_}\n")
    # print(f"PCA eigen vec[0]:{pca_vec[0]}")
    # print(f"PCA eigen vec[1]:{pca_vec[1]}\n")
    Em = pca_vec.T
    print(f'Dim Reduction [{N}x{N}] ---> [{N}x{M}]: \n {Em}\n')
    return Em

def Calc_X_exp(Em, X):
    # calculate the mean value of each Treasury (actually, each column is a treasury)
    mu = np.mean(X, axis=0)

    Rows, N = np.shape(X)
    # Detecting anomalies
    X_exp = []
    for rt_org in X:
        rt_exp = []
        for L in range(N):
            rt = rt_org.copy()
            rtL = rt[L]
            rt[L] = 0
            y = rt - mu
            Minus1 = np.zeros(N)  # .T
            Minus1[L] = -1
            x = np.c_[Em, Minus1]
            b = Calc_b(x, y)
            rtL_exp = b[-1]
            rt_exp.append(rtL_exp)
        X_exp.append(rt_exp)
    X_exp = np.array(X_exp)
    return X_exp

def Calc_dX(X, X_exp):
    # difference between input data to the expected values
    dX = abs(X-X_exp)
    # standartization the diff values (Normal distribution)
    dX = StandardScaler().fit_transform(dX)
    return dX

def Calc_th(dX):
    # np.mean(dX, axis=0)
    X_std = np.std(dX, axis=0)
    Rows = np.shape(dX)[0]
    th = np.tile(4*X_std, (Rows, 1))
    return th

def PlotS(X, X_exp, dX, th):
    plt.figure()
    plt.subplot(4,1,1)
    Plot(X, 'Treasury rates')
    plt.subplot(4,1,2)
    Plot(X_exp, 'Expected Treasury rates')
    plt.subplot(4,1,3)
    Plot(dX, 'residual (error)')
    plt.subplot(4,1,4)
    Plot(dX>th, 'anomaly')
    plt.ion()      
    plt.show()


def Plot(xxS, Str):
    labels = [str(i+1) for i in range(np.shape(xxS)[1])]
    for xx,label in zip(xxS.T,labels):
        plt.plot(np.arange(0,len(xx)), xx)#, label=label)
    plt.title(Str)
    # plt.legend()


def Calc_b(x,y):
    b = inv(np.dot(x.T,x))
    b = np.dot(b,x.T)
    b = np.dot(b,y)
    return b

MODE = 'Standartize InputS' # 'NO Standartize InputS'

# Read data
df = pd.read_csv('data.csv', header=None)
df.head()

# PreProcessing the data
X = df.iloc[:,1:].values
if 'Standartize InputS':
    X = StandardScaler().fit_transform(X)
else:
    pass


Em = calc_Em(X) # dimensionality reduction
X_exp = Calc_X_exp(Em, X)
dX = Calc_dX(X, X_exp)
th = Calc_th(dX)

PlotS(X, X_exp, dX, th)

anomalS = dX>th

# Print Anomalies:
# list(zip(*np.where(anomalS)))
for i, anomal in enumerate(anomalS.T):
    Treasury = i+1 # the first column in df is the date
    if any(anomal):
        print(f'Treasury{Treasury} has anomaly at:')
        df_i = df[anomal]
        for index, row in df_i.iterrows():
            Date, v = row[[0, Treasury]]
            print(f'{Date}: value={v}')
plt.ion()      
plt.show()


