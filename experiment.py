from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt 
import numpy as np

import sys
num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100

def get_ys(xs):
    signal = -0.1*xs**3 + xs**2 - 5*xs - 5
    noise = np.random.normal(0,100,(len(xs),1))
    return signal + noise

X = np.random.uniform(-20,20,num_samples).reshape((num_samples,1))
y = get_ys(X)

plt.scatter(X,y,label="data")

for degree in range(1,4):
    model = Pipeline([
        ("Poly", PolynomialFeatures(degree=degree)),
        ("LenReg", LinearRegression())
    ])
    model.fit(X,y)
    plotting_x = np.linspace(-20,20,num=50).reshape((50,1))
    preds = model.predict(plotting_x)
    plt.plot(plotting_x, preds, label=f"degree={degree}")

plt.legend()
plt.show()
plt.savefig("plot.png")

