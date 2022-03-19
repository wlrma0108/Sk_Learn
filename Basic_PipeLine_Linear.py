import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.style.use('seaborn')


model = LinearRegression(normalize=True)
print(model.normalize)
print(model)

x=np.arange(10)
y=2*x+1

print(x)
print(y)

X=x[:,np.newaxis]
print(X)
print(y)

model.fit(X,y)
print(model.coef_)
print(model.intercept_)

