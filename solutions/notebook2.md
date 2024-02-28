# Notebook 2 Solution

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
number_of_neighbors = 3
clf =  KNeighborsClassifier(number_of_neighbors)
clf.fit(X_train, y_train)
print(f"Accuracy is {round(100 * clf.score(X_test, y_test), 1)}")
```
