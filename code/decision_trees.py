### Assuming we have X, X_test, y, y_test from the exploratory_data_analysis.py

### Random Forests
### ----------------------------------------------------------------------------
### grid search
rf = RandomForestRegressor()  

rf_param_grid = {
    'bootstrap': [True],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 5],
    'n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=5, n_jobs=-1, verbose=0, scoring='neg_mean_squared_error')
grid_search.fit(X, y)  

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

### model fit on testing data
best_params = {
    'bootstrap': True,
    'max_depth': 20,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 300
}

rf_regressor_best = RandomForestRegressor(**best_params, random_state=42)
rf_regressor_best.fit(X, y)

y_pred = rf_regressor_best.predict(X_test)
mae = np.mean(abs(y_pred - y_test))
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'MAE: {mae}')
print("R-squared:", r_squared)
print("Mean Squared Error:", mse)

### XGBoost
### ----------------------------------------------------------------------------
### grid search

xgb_model = xgb.XGBRegressor()  
xgb_param_grid = {
    'max_depth': [3, 6, 9],  
    'learning_rate': [0.1, 0.01],  
    'n_estimators': [100, 200, 300],  
    'subsample': [0.8, 1.0],  
    'colsample_bytree': [0.8, 1.0],  
    'gamma': [0, 0.1],  
    'reg_alpha': [0, 0.1],  
    'reg_lambda': [0, 0.1]  
}
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)
grid_search_xgb.fit(X, y)  
print("XGBoost - Best parameters found: ", grid_search_xgb.best_params_)
print("XGBoost - Best negative mean squared error found: ", grid_search_xgb.best_score_)

### model fit on testing data
best_params = best_params = {
    'colsample_bytree': 0.8,
    'gamma': 0,
    'learning_rate': 0.1,
    'max_depth': 6,
    'n_estimators': 200,
    'reg_alpha': 0,
    'reg_lambda': 0.1,
    'subsample': 1.0
}

# Train a new model with the best parameters on the entire training dataset
best_xgb_model = xgb.XGBRegressor(**best_params)
best_xgb_model.fit(X, y)

# Evaluate the model on the test dataset
y_pred = best_xgb_model.predict(X_test)
mae = np.mean(abs(y_pred - y_test))
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'MAE: {mae}')
print("R-squared:", r_squared)
print("Mean Squared Error:", mse)
