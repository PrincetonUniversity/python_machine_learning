# Notebook 1 Solution

```python
laptop_X_train = laptop_X[:-20]
laptop_X_test = laptop_X[-20:]

laptop_y_train = laptop_y[:-20]
laptop_y_test = laptop_y[-20:]

regr.fit(laptop_X_train, laptop_y_train)

print("Intercept: \n", regr.intercept_)
print("Coefficients: \n", regr.coef_)
```
