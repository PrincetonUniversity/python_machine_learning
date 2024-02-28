# Notebook 3 Solution

```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2, min_samples=5, metric="euclidean")
y_db = db.fit_predict(X)
plt.scatter(X[y==0,0], X[y_db==0,1], c='blue')
plt.scatter(X[y==1,0], X[y_db==1,1], c='red')
```
