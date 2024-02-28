# Notebook 4 Solution

```python
from sklearn import manifold
digits = load_digits()
X = digits.data
y = digits.target
tsne = manifold.TSNE(n_components=2, init='pca', random_state = 0)
X_tsne = tsne.fit_transform(X)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.cm.nipy_spectral, edgecolor='k', label=y)
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
```
