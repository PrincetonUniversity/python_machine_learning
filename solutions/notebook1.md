# Notebook 1 Solution

```python
laptop_X_train = laptop_X[:-20]
laptop_X_test = laptop_X[-20:]

laptop_y_train = laptop_y[:-20]
laptop_y_test = laptop_y[-20:]

regr.fit(laptop_X_train, laptop_y_train)

print("Intercept: \n", regr.intercept_)
print("Coefficients: \n", regr.coef_)


laptop_y_pred_linear = regr.predict(laptop_X_test)

fig, axs = plt.subplots(figsize=(4.,4.), nrows=2, ncols=1, facecolor='white', dpi=200, sharex=True)
axs[0].scatter(laptop_X_train, laptop_y_train, s=1, color=qualitative_colors[0])
axs[0].plot(laptop_X_test, laptop_y_pred_linear, color=qualitative_colors[1], linewidth=2)
axs[0].scatter(laptop_X_test, laptop_y_test, color=qualitative_colors[2], s=8)
axs[1].hlines(0, 9, 20, color=qualitative_colors[1], linewidth=2)
axs[1].scatter(laptop_X_test, laptop_y_test-laptop_y_pred_linear, color=qualitative_colors[2], s=10)
axs[1].set_xlabel('Size in Inches')
axs[0].set_ylabel('Price (Euro)')
axs[1].set_ylabel('truth - model')
```
