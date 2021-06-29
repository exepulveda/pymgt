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
