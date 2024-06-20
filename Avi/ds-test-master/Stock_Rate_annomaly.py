import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
import random




def calc_E_M(X,M):
    pca = PCA(n_components=M)
    principalComponents = pca.fit_transform(X)
    
    E_M = pca.components_ # (M X N)

    # r_mean = df.mean() 
    r_cov = np.cov(X.T)

    # # Convert the covariance matrix to a numpy array
    # r_cov_np = r_cov.to_numpy()

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(r_cov)

    # check de-composition
    reconstructed_cov = np.matmul(eigenvectors, np.matmul(np.diag(eigenvalues), eigenvectors.T))
    # reconstructed_cov = eigenvectors@np.diag(eigenvalues)@np.diag(eigenvalues)@eigenvalues.T
    is_close = np.allclose(r_cov, reconstructed_cov)

    E = np.diag(eigenvalues)@eigenvectors.T

    # E_M = E[0:M,0:M]


    return E_M.T,principalComponents

def calc_X_M_exp(X,E_M):
    N = X.shape[1]
    X_M_exp = np.empty((0,N))
    for t in range(X.shape[0]):
        rt_pre = X[t,:]
        r_avg = np.mean(X,axis=0)
        rtl = np.empty((0, ))  # Assuming the rows have 4 columns      
        for L in range (N):
            rt = np.copy(rt_pre)
            rt[L] = 0
            y = rt-r_avg            
            minus_1_vec = np.zeros((N,1))
            minus_1_vec[L] = -1
            X1 = np.concatenate((E_M,minus_1_vec),axis=1)
            beta = np.linalg.inv((X1.T@X1))@X1.T@y
            rtl = np.append(rtl, beta[-1])
        X_M_exp = np.append(X_M_exp,rtl.reshape(1,-1),axis = 0)
    return X_M_exp

def Plot(xxS, Str):
    labels = [str(i+1) for i in range(np.shape(xxS)[1])]
    for xx,label in zip(xxS.T,labels):
        plt.plot(np.arange(0,len(xx)), xx)#, label=label)
    plt.title(Str)
    # plt.legend()



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
    plt.show(block=True)



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



# read the data
gen_data = True
M = 2
np.random.seed(10)
# X_reconstructed = pca.inverse_transform(X_pca)

if (gen_data):
    df = pd.DataFrame()
    N_features = 4
    N_outliers = 0
    N_samples = 100
    mean_vec = [0,0,0,0]
    std_vec = [10,1,1,0]
    outlier_ind = np.random.randint(0,N_samples,size=N_outliers)

    for i in range(N_features):
        df[f'f{i}'] = np.random.normal(loc=mean_vec[i],scale=std_vec[i], size=N_samples)
        df[f'f{i}'].loc[outlier_ind]=df[f'f{i}'].loc[outlier_ind]*10      
    df.plot(x='f0',y='f1',kind='scatter',xlim=[-30,30],ylim=[-30,30])
    plt.ion()      
    plt.show()
    
else:
    df = pd.read_csv('./data.csv')
    df.set_index('19-10-17',inplace=True)

# pre-process
X = StandardScaler().fit_transform(df.loc[:,:])
plt.plot()

E_M,principalComponents = calc_E_M(X,M)

X_M_exp = calc_X_M_exp(X,E_M)
if (gen_data):
    plt.figure()
    plt.scatter(X_M_exp[:,0],X_M_exp[:,1])
    plt.xlim([-30,30])
    plt.ylim([-30,30])
    plt.ion()      
    plt.show()
 
dx = Calc_dX(X,X_M_exp)

th = Calc_th(dx)
PlotS(X,X_M_exp,dx,th)

