#%%
import matplotlib

# %%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import make_blobs
plt.style.use('seaborn')

X,y=make_blobs(n_samples=50,centers=2,random_state=0, cluster_std=0.60)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='spring');

xfit= np.linspace(-1,3.5)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='spring');

for m,b in [(1.0,65),(0.5,1.6),(-0.2,2.9)]:
    plt.plot(xfit, m*xfit+b,'-k')

plt.xlim(-1,3.5);
plt.show()
