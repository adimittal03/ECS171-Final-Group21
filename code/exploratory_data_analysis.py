# import necessary modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import SGD 
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.regularizers import l1, l2

df = pd.read_csv("~/Desktop/Year_three/Spring_2024/ECS171/ECS171_Final/diamonds.csv") 
df = df.iloc[:, 1:]  # remove index column           
df.head()

### EXPLORATORY DATA ANALYSIS
### Plot distribution of continous features 
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

### Plot Histogram frequency of categorical variables
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

### Correlation Matrix between continous predictors and Y
correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Continuous Features')
plt.show()

### Pairwise Scatterplots
feature_names = ['carat', 'depth', 'table', 'x','y','z']
y = df['price']
plt.figure(figsize=(15, 10)) # Create scatter plots
for i, feature in enumerate(feature_names):
    plt.subplot(3, 2, i+1)
    plt.scatter(df[feature], y, alpha=0.5, s=15)  # s=20 sets the size of the points
    plt.title(f'{feature} vs price')
    plt.xlabel(feature)
    plt.ylabel('price')
plt.tight_layout()
plt.show()

### DATA Pre-processing 
### -----------------------------------------------------------------------------
# Label encodings // Custom label encoding mappings
cut_mapping = {
    'Fair': 0,
    'Good': 1,
    'Very Good': 2,
    'Premium': 3,
    'Ideal': 4
}

color_mapping = {
    'J': 0,
    'I': 1,
    'H': 2,
    'G': 3,
    'F': 4,
    'E': 5,
    'D': 6
}

clarity_mapping = {
    'I1': 0,
    'SI2': 1,
    'SI1': 2,
    'VS2': 3,
    'VS1': 4,
    'VVS2': 5,
    'VVS1': 6,
    'IF': 7
}

# Apply the mappings to the dataset
df['cut_encoded'] = df['cut'].map(cut_mapping)
df['color_encoded'] = df['color'].map(color_mapping)
df['clarity_encoded'] = df['clarity'].map(clarity_mapping)
df = df.drop(['cut', 'color', 'clarity'], axis=1)

# creating X and y
y = df['price']
df.drop(columns=['price'], inplace=True)
X = df

### Standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
scaled_df.head()

### removing outliers from carat
q1 = scaled_df['carat'].quantile(0.25)
q3 = scaled_df['carat'].quantile(0.75)
iqr = q3 - q1 
lower_bound = q1 - 1.5 * iqr 
upper_bound = q3 + 1.5 * iqr
scaled_df_no_outliers = scaled_df[(scaled_df['carat'] > lower_bound) & (scaled_df['carat'] < upper_bound)]
y = y[(scaled_df['carat'] > lower_bound) & (scaled_df['carat'] < upper_bound)]

# outlier box-plots
num_rows = 3
num_cols = 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
axes = axes.flatten()
for i, feature in enumerate(features_to_plot):
    sns.boxplot(x=scaled_df_no_outliers[feature], ax=axes[i])
    axes[i].set_title(f'Boxplot of {feature}')
for j in range(i + 1, num_rows * num_cols):
    axes[j].axis('off')
plt.tight_layout()
plt.show()

### This is our X and y from cleaned dataset - 51,800 rows
X = scaled_df_no_outliers
y = y
