### Assuming we have X, X_test, y, y_test from exploratory_data_analysis.py

### Linear Regression
### ----------------------------------------------------------------------------

### Cross validation procedure
model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42) 

mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
mse_scores = -mse_scores
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)
print(f'Mean MSE: {mean_mse}')
print(f'Standard Deviation of MSE: {std_mse}')


### Model accuracy on testing data
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X_test)
mae = np.mean(abs(y_pred - y_test))
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'MAE: {mae}')
print("R-squared:", r_squared)
print("Mean Squared Error:", mse)


### Polynomial Regression
### ----------------------------------------------------------------------------
poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(X)
X_poly_test = poly.fit_transform(X_test)

# Cross validation procedure
model = LinearRegression()
kf = KFold(n_splits=10, shuffle=True, random_state=42) # 5 fold CV
mse_scores = cross_val_score(model, X_poly, y, cv=kf, scoring='neg_mean_squared_error')
mse_scores = -mse_scores
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)
print(f'Mean MSE: {mean_mse}')
print(f'Standard Deviation of MSE: {std_mse}')

# model fit on testing data
model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly_test)
mae = np.mean(abs(y_pred - y_test))
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'MAE: {mae}')
print("R-squared:", r_squared)
print("Mean Squared Error:", mse)
