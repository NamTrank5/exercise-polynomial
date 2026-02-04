# Polynomial Regression Example using MLflow projects

We’ve made a small project that shows an example of polynomial regression, which we will use to show how MLflow projects work. The main experiment, in which we simulate some data and model it using different degrees of polynomial regression, is defined in `experiment.py`:

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
import numpy as np

import sys
num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100

def get_ys(xs)
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
```

The pip environment for this experiment can be specified using a python_env.yaml file:

```bash
# Dependencies required to build packages. This field is optional.
build_dependencies:
  - pip
# Dependencies required to run the project.
dependencies:
  - scipy
  - scikit-learn>0.23
  - numpy>1.19
  - mlflow
  - matplotlib>3
```

Finally, the MLproject-file specifies how to run the project:

```bash
name: PolyReg

python_env: python_env.yaml
# or
# conda_env: my_env.yaml
# or
# docker_env:
#    image:  mlflow-docker-example

entry_points:
  main:
    parameters:
      num_samples: {type: int, default: 100}
    command: "python experiment.py {num_samples}"
```

With these three things in place, you can run the experiment using `mlflow run <path to project>`. So if if you are located in the project directory:

```bash
mlflow run .
```

Because this project is hosted as a git repository, you can simplxy do:

```bash
mlflow run https://github.com/LSDA-BDM/exercise-polynomial.git
```

This will fetch the project, resolve the environment, and run the main entry point with the default parameters.
If you want to run the experiment with 500 samples, instead of the default 100, you can do:

```bash
mlflow run https://github.com/LSDA-BDM/exercise-polynomial.git -P num_samples=500
```
