# Python Multivariate Gaussian Transforms: pymgt

An implementation of several useful multivariate Gaussian transforms (MGT)
in the context of geostatistics, but usable in many other contexts.

Any MGT will be able to transform a multivariate data `x` into a new set of 
uncorrelated factors `y` that follow a standard Gaussian distribution with unit covariance:
`y` such as `y = MGT.transform(x) and x = MGT.inverse_transofrm(y)`.

The implemented MGT are:
- Rotation based iterative Gaussianisation (RBIG)
- Projection pursuit multivariate transform (PPMT)

## Rotation based iterative Gaussianisation

This transform iteratively uses the independent component analysis (ICA) to rotate the data
and applies gaussianisation to the marginals.

Reference:
Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative gaussianization: From ICA to random rotations. IEEE Transactions on Neural Networks, 22(4), 537–549. https://doi.org/10.1109/TNN.2011.2106511

## Projection pursuit multivariate transform

This transform iteratively gaussianises one projection found by maximising the Friedman index for
departure from Gaussian. It can use the original gradient descend optimisation method (fastest)
or differential evolution (slower but more robust).

Reference:
Barnett, R. M., Manchuk, J. G., & Deutsch, C. V. (2013). Projection Pursuit Multivariate Transform. Mathematical Geosciences, 46(3), 337–359. https://doi.org/10.1007/s11004-013-9497-7

## How to install

With pip, just execute: `python -m pip install pymgt`

## Simple use

```python
import matplotlib.pyplot as plt
from pymgt import PPMTransform
from pymgt import RBIGTransform
from pymgt import DEFAULT_METRICS
from sklearn.datasets import make_moons

x, _ = make_moons(n_samples=1000, noise=0.05)

ppmt = PPMTransform(target=0.001, metrics=DEFAULT_METRICS) # with default parameters

y_ppmt = ppmt.fit_transform(x)
x_back_ppmt = ppmt.inverse_transform(y_ppmt)

rbigt = RBIGTransform(target=0.001, metrics=DEFAULT_METRICS) # with default parameters

y_rbig = rbigt.fit_transform(x)
x_back_rbig = rbigt.inverse_transform(y_rbig)

# plot scatter plot
fig, ax = plt.subplots(1, 5, figsize=(20, 5), tight_layout=True)

ax[0].scatter(x[:, 0], x[:, 1])
ax[1].scatter(y_ppmt[:, 0], y_ppmt[:, 1])
ax[2].scatter(x_back_ppmt[:, 0], x_back_ppmt[:, 1])
ax[3].scatter(y_rbig[:, 0], y_rbig[:, 1])
ax[4].scatter(x_back_rbig[:, 0], x_back_rbig[:, 1])

for i in range(5):
  if i in [0, 2, 4]:
    ax[i].set_xlim(-1.5, 2.5)
    ax[i].set_ylim(-0.8, 1.5)
  else:
    ax[i].set_xlim(-3.5, 3.5)
    ax[i].set_ylim(-3.5, 3.5)
  
  ax[i].set_xlabel("X1")
  ax[i].set_ylabel("X2")

ax[0].set_title("Original data")
ax[1].set_title("Gaussian by PPMT")
ax[2].set_title("Back transformed by PPMT")
ax[3].set_title("Gaussian by RBIG")
ax[4].set_title("Back transformed by RBIG")

plt.show()
```

## How to test

The test suite relays on the *pytest* package. Install it if needed:
```python -m pip install pytest```

To run the test suite, execute
```pytest```

## Acknowledgements

The first version of this package was part of the research
conducted by the Australian Research Council Integrated Operations for Complex
Resources Industrial Transformation Training Centre (project number IC190100017)
and funded by the Australian Government.
