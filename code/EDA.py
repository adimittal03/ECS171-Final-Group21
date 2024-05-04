# import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

### DATA-Preprocessing & EDA Code

# load data
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_directory)
os.chdir(parent_dir)
print(parent_dir)
df = pd.read_csv("diamonds.csv") 
df = df.iloc[:, 1:]               # remove index column
# df.head()

### ----------------------------------------------------------------------------

### Plot distribution of each continous feature
features_to_plot = ['carat', 'depth', 'table', 'x','y','z']
plt.figure(figsize=(8, 6))
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[feature], kde=True)
    plt.title(feature)

plt.tight_layout()
plt.show()
print("\nmean:\n{} , \nmedian:\n{} , \nstd:\n {}".format(df[features_to_plot].mean() , 
                                                         df[features_to_plot].median(), 
                                                         df[features_to_plot].std()))

### ----------------------------------------------------------------------------

### Frequency of categorical vars
categorical_vars = ["cut", "color", "clarity"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, var in enumerate(categorical_vars):
    sns.countplot(data=df, x=var, ax=axes[i])
    axes[i].set_title("Frequency of " + var)
    axes[i].set_xlabel(var)
    axes[i].set_ylabel("Frequency")
    axes[i].tick_params(axis='x', rotation=45)  
plt.tight_layout()
plt.show()

### ----------------------------------------------------------------------------

### Correlation Matrix between continous predictors and Y
correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Continuous Features')
plt.show()


### ----------------------------------------------------------------------------

### Plot the relationship of each continous feature vs Y
continuous_vars = ['carat', 'depth', 'table', 'x','y','z']
y_variable = 'price'
num_plots = len(continuous_vars)
num_cols = 3  # Number of plots per row
num_rows = (num_plots + num_cols - 1) // num_cols
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
for i, var in enumerate(continuous_vars):
    row = i // num_cols
    col = i % num_cols
    sns.scatterplot(data=df, x=var, y=y_variable, ax=axes[row, col], s = 5)
    axes[row, col].set_title(f"{var} vs {y_variable}")
    axes[row, col].set_xlabel(var)
    axes[row, col].set_ylabel(y_variable)

# Adjust layout
plt.tight_layout()
plt.show()

### ----------------------------------------------------------------------------

### LABEL ENCODINGS
mapping1 = {'I1': 0.0, 'SI2': 1.0, 'SI1': 2.0, 'VS2': 3.0, 'VS1':4.0, 'VVS2':5.0, 'VVS1':6.0, 'IF':7.0}
mapping2 = {'D': 6.0, 'E': 5.0, 'F': 4.0, 'G':3.0, 'H':2.0,'I':1.0,'J':0.0}
mapping3 = {'Fair': 0.0, 'Good': 1.0, 'Very Good': 2.0, 'Premium':3.0 ,'Ideal': 4.0}
df['clarity'] = df['clarity'].map(mapping1)
df['color'] = df['color'].map(mapping2)
df['cut'] = df['cut'].map(mapping3)
df = pd.DataFrame(df)

# creating X and y
y = df['price']
df.drop(columns=['price'], inplace=True)
X = df

# scaling
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(X))
scaled_df.columns = X.columns
scaled_df.head()

### ----------------------------------------------------------------------------

### removing outliers from carat
q1 = scaled_df['carat'].quantile(0.25)
q3 = scaled_df['carat'].quantile(0.75)
iqr = q3 - q1 
lower_bound = q1 - 1.5 * iqr 
upper_bound = q3 + 1.5 * iqr
scaled_df_no_outliers = scaled_df[(scaled_df['carat'] > lower_bound) & (scaled_df['carat'] < upper_bound)]
y = y[(scaled_df['carat'] > lower_bound) & (scaled_df['carat'] < upper_bound)]

### ----------------------------------------------------------------------------

for i in features_to_plot:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=scaled_df_no_outliers[i])
    plt.title(f'Boxplot of {i}')
    plt.show()
