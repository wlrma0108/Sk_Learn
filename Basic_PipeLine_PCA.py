#%%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from __future__ import print_function, division
from sklearn.datasets import load_digits

plt.style.use('seaborn')
np.random.seed(1)
X=np.dot(np.random.random(size=(2,2)),np.random.normal(size=(2,200))).T
plt.plot(X[:,0],X[:,1],'o')
plt.axis('equal');
pca=PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_)
print(pca.components_)
plt.plot(X[:,0],X[:,1],'o',alpha=0.5)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v=vector *3 *np.sqrt(length)
    plt.plot([0,v[0]],[0,v[1]],'-k',lw=3)
plt.axis('equal');


clf=PCA(0.95)
X_trans=clf.fit_transform(X)
print(X.shape)
print(X_trans.shape)

X_new=clf.inverse_transform(X_trans)
plt.plot(X[:,0],X[:,1],'o',alpha=0.2)
plt.plot(X_new[:,0],X_new[:,1],'ob',alpha=0.8)
plt.axis('equal');

digits=load_digits()
X=digits.data
y=digits.target

pca=PCA(2)
Xproj=pca.fit_transform(X)
print(X.shape)
print(Xproj.shape)

plt.scatter(Xproj[:,0],Xproj[:,1],c=y,edgecolor='none',alpha=0.5,cmap=plt.cm.get_cmap('nipy_spectral',10))
plt.colorbar();



