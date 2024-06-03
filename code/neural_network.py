### Assuming we have X, X_test, y, y_test from exploratory_data_analysis.py

### Defining the Neural Network ------------------------------------------------
### Grid Search to find best hyper-params
def create_model(num_layers = 3, num_neurons = 64, dropout_rate=0.0, optimizer='adam', learning_rate = 0.01):
    model = Sequential()

    # hidden layer one
    model.add(Dense(num_neurons, input_dim=9, activation='relu'))  
    model.add(Dropout(dropout_rate))
    
    # following hidden layers
    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # output layer
    model.add(Dense(1, activation='relu'))
    
    # optimizer
    if optimizer == 'rmsprop':
        opt = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'adam':
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        opt = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
    else:
        raise ValueError('optimizer {} unrecognized'.format(optimizer))
        
    # compile
    network.compile(optimizer = opt, 
                loss = 'mean_squared_error', 
                metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    
    return model

### Implement Grid Search -------------------------------------------------------
ann = KerasRegressor(build_fn=create_model, verbose=0)

# these are the values for the other hyperparameters
param_grid = dict(
    num_layers= [1, 3, 5, 8],
    num_neurons= [16, 32, 64],
    dropout_rate= [0.2, 0.5],
    optimizer = ['rmsprop', 'adam', 'SGD'],
    batch_size = [32, 64, 128, 256, 512, 1024, 2048],
    epochs = [10, 25, 50, 100, 150, 200],
    learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
)

# apply 5-fold CV to grid search
grid = GridSearchCV(estimator=ann, cv=5, param_grid=param_grid)
grid_result = grid.fit(X, y)

# print best set of parameters // comes after the warnings 
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
